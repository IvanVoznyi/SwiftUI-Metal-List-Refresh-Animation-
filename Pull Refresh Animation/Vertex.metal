//
//  Vertex.metal
//  Pull Refresh Animation_V2
//
//  Created by Ivan Voznyi on 2/9/26.

#include <metal_stdlib>
using namespace metal;

struct Particle {
    float2 position;
    float4 color;
    float size;
};

struct Uniforms {
    float2 screenSize;
    float scrollOffset;
    float time;
};

struct Parameters {
    float4 color;
    int particleCount;
    float triangleTop;
    float triangleBottom;
    float opacityInCircle;
    float glowIntensity;
};

struct VertexOut {
    float4 position [[position]];
    float4 color;
    float size [[point_size]];
    float mode;
};

static float hash(float n) {
    return fract(sin(n) * 43758.5453123);
}

// --- MASTER SIZES ---
constant float FORMATION_RADIUS      = 55.0;
constant float DOT_SIZE_TARGET       = 15.0;
constant float DOT_SIZE_FALLING_BASE = 10.0;
constant float DOT_SIZE_VARIANCE     = 1.8;
constant float PARTICLE_BAND_WIDTH   = 1.0;

// --- TIMING & SCROLL ---
constant float SCROLL_NORM_FACTOR    = 75.0; // The refresh threshold should match the standard SwiftUI refresh threshold "thresholdRefresh variable".
constant float MORPH_START_PCT       = 0.85;
constant float RING_APPEAR_START_PCT = 0.95;
constant float CIRCLE_DETECT_LIMIT   = 0.85;

// --- PHYSICS (FALLING FUNNEL) ---
constant float VERTICAL_SPREAD_START = -50.0;
constant float VERTICAL_SPREAD_DEPTH = 350.0;
constant float FALL_STAGGER_OFFSET   = 0.4;
constant float FALL_STAGGER_SCALE    = 0.6;
constant float SPEED_VARIATION       = 4.0;

// --- TURBULENCE (WIGGLE) ---
constant float TURB_FREQ_X           = 7.0;
constant float TURB_AMP_X            = 8.0;
constant float TURB_FREQ_Y           = 4.5;
constant float TURB_AMP_Y            = 3.5;

// --- PULSE ANIMATION ---
constant float PULSE_SPEED           = 4.0;
constant float PULSE_INTENSITY       = 0.05;

// --- VISUALS ---
constant float OPACITY_FALLING       = 1.0;
constant float GLOW_FALLOFF_POWER    = 1.0;
constant float RING_GLOW_SPREAD      = 4.0;
constant float RING_CORE_THINNING    = 0.4;

// --- AUTO-CALCULATED ---
constant float SOLID_CANVAS_MULTIPLIER = 5.0;
constant float SOLID_CANVAS_SIZE       = FORMATION_RADIUS * SOLID_CANVAS_MULTIPLIER;
constant float ANTIALIAS_SOFTNESS      = 0.01;

// -- RADIAL BOUNCE (Spring Effect) --

// As they arrive, they compress the radius slightly and snap back.
constant float BOUNCE_FREQ = 15.0; // Fast vibration
constant float BOUNCE_AMP = 8.0;   // Pixels of bounce

// -- WIGGLE --
constant float BASE_WIGGLE_FREQ = 12.0;
constant float WIGGLE_VARIATION = 0.1;
constant float WIGGLE_AMP = 15.0;

// --- Constants (Place at the top of your shader file) ---
constant float SIZE_TIER_LARGE_THRESHOLD = 0.4;
constant float SIZE_TIER_MEDIUM_THRESHOLD = 0.8;
constant float SIZE_SCALE_LARGE  = 1.0;
constant float SIZE_SCALE_MEDIUM = 0.8;
constant float SIZE_SCALE_SMALL  = 0.5;

//--- The Compute Kernel (The Physics Engine) ---
// This function runs in parallel for every single particle in the system.
// It is the "brain" responsible for motion, forces, and lifecycle management.
// 1. [[buffer(0)]] particles:
//    - The "Master Ledger" of our simulation.
//    - Since it is in the `device` address space, we can Read AND Write to it.
//    - We read the current position/velocity, apply physics math, and
//      save the new values back to this same memory slot.
// 2. [[buffer(1)]] uniforms:
//    - The "Environment" variables.
//    - Contains data that changes every frame but is consistent for all particles,
//      such as Delta Time (dt), the current time, or the mouse position.
// 3. [[buffer(2)]] parameters:
//    - The "Rulebook" or "Tuning Knobs."
//    - Static settings defined by the user, such as gravity strength,
//      drag/friction, or the maximum speed limit of the liquid.
// 4. [[thread_position_in_grid]]:
//    - The "ID Badge" for this specific thread.
//    - If we launch 10,000 threads, this number (id) ranges from 0 to 9,999.
//    - We use this index to ensure this specific thread only touches
//      the data at `particles[id]`.
kernel void updateParticles(
                            device Particle *particles [[buffer(0)]],
                            constant Uniforms &uniforms [[buffer(1)]],
                            constant Parameters &parameters [[buffer(2)]],
                            uint id [[thread_position_in_grid]])
{
    // --- Normalizing the Pull Strength (The "Intensity" Factor) ---
    // Converts the raw scroll distance into a clean 0.0 to 1.0 range.
    // 1. The Normalization:
    //    - We divide the current `scrollOffset` (e.g., 150.0) by the `SCROLL_NORM_FACTOR`.
    //    - This tells us "How much of the total pull-to-refresh distance has been covered?"
    //      (e.g., 0.5 means the user is halfway through the pull).
    // 2. The `saturate` Function (The GPU's Best Friend):
    //    - In Metal, `saturate(x)` is a hardware-optimized shortcut for `clamp(x, 0.0, 1.0)`.
    //    - It is incredibly fast. Even if the user pulls their finger way past the limit,
    //      `pull` will never exceed 1.0. This prevents the liquid physics from
    //      "exploding" or stretching into infinity.
    // 3. The Foundation for Motion:
    //    - Every physics calculation that follows—the wobbling, the stretching, and
    //      the gooey tension—is driven by this single `pull` value.
    float pull = saturate(uniforms.scrollOffset / SCROLL_NORM_FACTOR);
    // --- Defining the Morphing Window (The Animation Trigger) ---
    // Controls exactly when the liquid begins its transition from "Stable" to "Stretchy."
    // 1. The Delayed Start (MORPH_START_PCT):
    //    - We don't want the liquid to start deforming the micro-second the user
    //      starts pulling.
    //    - By using `MORPH_START_PCT` (e.g., 0.3), we ensure nothing happens for
    //      the first 30% of the pull. This creates a "tension" feel, where the
    //      liquid resists until a certain threshold is met.
    // 2. The S-Curve Transition:
    //    - Using `smoothstep` here ensures that once the morphing *does* start,
    //      it accelerates smoothly.
    //    - `morphPhase` goes from 0.0 to 1.0, but it does so with a soft "Ease-In,"
    //      making the gooey stretching feel organic rather than linear and robotic.
    // 3. Driving the Shader Logic:
    //    - This value will later be used to interpolate between the "Resting" state
    //      of the particles and their "Stretched" state.
    float morphPhase = smoothstep(MORPH_START_PCT, 1.0, pull);
    // --- The Core Visibility (The "Solid Ring" Reveal) ---
    // Manages the transparency of the central liquid body as the user pulls.
    // 1. Defining the Appearance Threshold:
    //    - Similar to the morph phase, we use `RING_APPEAR_START_PCT` to delay the
    //      visibility. The "Solid Ring" (the base of our pull-to-refresh) only starts
    //      fading in after the pull reaches a specific percentage.
    // 2. Ensuring a Soft Entrance:
    //    - By mapping the `pull` to an alpha value (0.0 to 1.0) through `smoothstep`,
    //      we avoid a harsh "pop-in." Instead, the liquid appears to coalesce
    //      from the background, becoming fully opaque exactly as the pull hits 100%.
    // 3. Visual Layering:
    //    - This alpha value allows us to separate the "gooey" particle physics from
    //      the "solid" central body. It gives the user a visual anchor to focus on
    //      while the more chaotic particles dance around the edges.
    float solidRingAlpha = smoothstep(RING_APPEAR_START_PCT, 1.0, pull);
    // --- Indexing the Anchor (The Ring Reference) ---
    // Specifically targets the very last particle in our buffer to act as a
    // structural reference point.
    // 1. Array-Based Logic:
    //    - Since programming arrays are "Zero-Indexed," the position of the last
    //      item is always `Total Count - 1`.
    //    - If we have 250 particles, this index is 249.
    // 2. The Anchor Role:
    //    - In this simulation, most particles are "free-floating" to create the
    //      gooey liquid effect. However, we often use the last particle as a
    //      stationary "pole" or "ring" around which the others rotate or attract.
    // 3. Type Conversion:
    //    - We explicitly cast the result to a `uint` (unsigned integer) to match
    //      Metal's indexing requirements, ensuring the GPU looks at a precise
    //      memory address without any floating-point ambiguity.
    uint ringIndex = uint(parameters.particleCount - 1);
    // --- Floating-Point Normalization (The Range Mapping) ---
    // Converts the integer index into a float to allow for smooth division and math.
    // 1. Enabling Precision:
    //    - We can't perform "smooth" math with integers (like 249 / 2). By casting
    //      the `ringIndex` to a `float`, we enable the GPU to calculate fractional
    //      positions and percentages.
    // 2. The Normalization Base:
    //    - This value represents the maximum "step" in our particle list.
    //    - We will use this as a denominator to give every particle a unique
    //      percentage (e.g., Particle #125 / 250 = 0.5), which allows us to
    //      arrange them in a perfect circle or line later on.
    // 3. Mathematical Consistency:
    //    - In Shaders, mixing types (int vs float) can lead to compilation errors
    //      or unexpected "stepping" artifacts. Pre-converting this to a float
    //      ensures our trig functions (sin/cos) behave predictably.
    float totalDots = float(ringIndex);
    
    // SOLID RING
    
    // --- The Structural Branch (Isolating the Anchor) ---
    // Distinguishes between the "free" particles and the stationary "foundation" particle.
    // 1. Identifying the Ring:
    //    - If the current thread's `id` matches or exceeds the `ringIndex` (249),
    //      it means the GPU is currently processing the very last particle in our buffer.
    // 2. Specialized Behavior:
    //    - While the other 249 particles are off "simulating" liquid physics,
    //      this specific particle is treated differently. It acts as the "center of mass"
    //      or the visual core that stays put while everything else stretches around it.
    // 3. Conditional Optimization:
    //    - In Shaders, branching (`if` statements) should be used carefully. Here,
    //      it’s essential because it allows us to give the "Ring" particle its
    //      own unique properties—like transparency and position—without affecting
    //      the physics of the liquid droplets.
    if (id >= ringIndex) {
        // --- Centering the Anchor (The Zero-Point Calibration) ---
        // Places the foundation particle at the exact geometric center of the screen.
        // 1. Defining the Origin:
        //    - By dividing the `screenSize` (width and height) by 2.0, we find the
        //      precise midpoint of the coordinate system.
        //    - This ensures that no matter what iPhone model or screen size is
        //      being used, the "liquid source" is always perfectly balanced.
        // 2. The Anchor Position:
        //    - This specific particle acts as the "Fixed Point" for our animation.
        //      While other particles will stretch downwards during the pull,
        //      this one remains the stable pivot that the user’s eye tracks.
        // 3. Vector Math:
        //    - Since `position` and `screenSize` are both 2D vectors (float2),
        //      dividing by 2.0 is a single, lightning-fast SIMD operation that
        //      scales both X and Y simultaneously.
        particles[id].position = uniforms.screenSize / 2.0;
        // --- Creating the Breath (The Sinusoidal Pulse) ---
        // Generates a rhythmic scaling factor to make the liquid feel "alive."
        // 1. The Sine Wave Foundation:
        //    - We use `sin(time * speed)` to create a value that oscillates
        //      smoothly between -1.0 and 1.0 over time.
        // 2. Mapping the Intensity:
        //    - By multiplying the wave by `PULSE_INTENSITY` (e.g., 0.05), we
        //      shrink that range so the movement is subtle rather than violent.
        //    - Adding `1.0` shifts the range to roughly 0.95 to 1.05. This ensures
        //      the liquid "breathes" around its natural size rather than
        //      disappearing into a negative scale.
        // 3. The Visual "Heartbeat":
        //    - This value is applied to the ring or the base of the pull-to-refresh.
        //      It provides a small amount of visual feedback even when the user
        //      isn't moving their finger, signaling that the simulation is
        //      active and "waiting" for the refresh to trigger.
        float pulse = 1.0 + PULSE_INTENSITY * sin(uniforms.time * PULSE_SPEED);
        // --- Setting the Anchor Color (The Theme Injection) ---
        // Finalizes the appearance of the core ring by applying the color and transparency.
        // 1. Vector Composition (float4):
        //    - We take the RGB channels from our `uniforms.color` (the base theme)
        //      and pair them with the `solidRingAlpha`.
        //    - This creates a complete 4-component vector (Red, Green, Blue, Alpha)
        //      that the Fragment Shader uses to paint the pixel.
        // 2. Dynamic Transparency:
        //    - By using `solidRingAlpha` here, the center of the pull-to-refresh
        //      smoothly fades in as the user pulls down.
        //    - This ensures that when the "pull" is just starting, the ring is
        //      invisible, appearing only as the "gooey" tension builds up.
        // 3. Maintaining Visual Consistency:
        //    - Using the RGB from the uniforms ensures that the "solid" part of
        //      the liquid matches the color of the individual droplets exactly,
        //      maintaining the illusion that this is one continuous, viscous fluid.
        particles[id].color = float4(parameters.color.rgb, solidRingAlpha);
        // --- Scaling the Foundation (The Dynamic Core Size) ---
        // Determines the physical footprint of the anchor particle on the screen.
        // 1. Base Dimension (SOLID_CANVAS_SIZE):
        //    - We start with a constant value that defines the "resting" size of
        //      the central liquid body. This ensures the core is large enough
        //      to act as the visual heart of the animation.
        // 2. Multiplying the Breath:
        //    - By multiplying the base size by our `pulse` factor (e.g., 0.95 to 1.05),
        //      the anchor physically grows and shrinks in sync with the sine wave.
        //    - This "organic scaling" makes the liquid feel pressurized and active,
        //      as if it is bulging slightly with internal tension.
        // 3. Pixel Mapping:
        //    - This value is passed to the Render Pipeline as `[[point_size]]`.
        //      The GPU uses this to expand our single coordinate into a square
        //      rasterization area, which our Fragment Shader then rounds into
        //      a smooth, gooey circle.
        particles[id].size = SOLID_CANVAS_SIZE * pulse;
        return;
    }
    
    // DOTS
    
    // --- The Identity Conversion (Integer to Float) ---
    // Casts the unique thread ID into a floating-point number for math operations.
    // 1. Why the Cast is Necessary:
    //    - The system gives us `id` as a `uint` (0, 1, 2...). However, physics
    //      formulas like sine waves and division require `float` types.
    //    - If we didn't cast this, `id / totalDots` would perform "Integer Division"
    //      (truncating decimals), causing all our particles to snap to grid lines
    //      instead of moving smoothly.
    // 2. The Basis for Variation:
    //    - This value `i` is the DNA of the particle. We will use it to give
    //      every single dot a slightly different offset, speed, and angle so
    //      they don't all move in a rigid block.
    float i = float(id);
    // --- The Horizontal Randomness (The X-Axis Seed) ---
    // Generates a unique, deterministic random value for this particle's X position.
    // 1. The Hash Function:
    //    - Unlike `rand()`, a hash is deterministic: Inputting the same number
    //      always yields the same result. This is crucial for GPU physics so the
    //      liquid doesn't "jitter" or change shape every frame.
    // 2. The Offset (+ 12.0):
    //    - We add `12.0` to the particle ID `i` to create a unique "seed" for
    //      the X-axis.
    //    - This ensures that the random value we get here is mathematically
    //      independent from the Y-axis or Size randomness we'll calculate later.
    // 3. Usage:
    //    - We store this in `hX`. We will use this value to slightly offset the
    //      particle to the left or right, preventing the liquid from looking like
    //      a perfect, artificial grid.
    float hX = hash(i + 12.0);
    // --- The Vertical Variance (The Y-Axis Seed) ---
    // Generates a second, distinct random value for the particle's Y-axis properties.
    // 1. Avoiding Correlation:
    //    - If we used the same seed (e.g., `i + 12.0`) for both X and Y, `hX` would
    //      equal `hY`. This would cause every particle to line up perfectly on a
    //      diagonal line (x = y), looking artificial.
    // 2. The Offset (+ 24.0):
    //    - By adding a different arbitrary number (24.0), we feed a different input
    //      into the deterministic hash. This guarantees a result that is mathematically
    //      unrelated to `hX`.
    // 3. The Result:
    //    - We now have two independent "dice rolls" for this particle. We can use `hX`
    //      to scatter it left/right and `hY` to scatter it up/down, creating a
    //      truly organic 2D distribution.
    float hY = hash(i + 24.0);
    // --- The Size Variance (The Size Dot Seed) ---
    // Generates a third, independent random value for the particle's scale.
    // 1. Independent Axis:
    //    - Just as X and Y needed separate seeds, we need a unique seed for Size
    //      (hence 'S'). If we re-used hX or hY, all "left-side" particles or
    //      "top-side" particles would be the same size, creating visible patterns.
    // 2. The Offset (+ 67.0):
    //    - We add a distinct arbitrary number (67.0) to `i`. This ensures `hS`
    //      is uncorrelated with the position.
    // 3. The Result:
    //    - We now have a "Size DNA." We can use this to make some droplets tiny
    //      (0.5x scale) and others large and heavy (1.5x scale), giving the liquid
    //      texture and depth instead of looking like uniform polka dots.
    float hS = hash(i + 67.0);
    // --- The Velocity Variance (The "Speed" DNA) ---
    // Generates a unique random value specifically for the particle's movement speed.
    // 1. The Base Seed:
    //    - Notice we use just `i` here (no +12.0 or +67.0). This is our "base" hash.
    //    - Since we used offsets for position and size, using the raw index here
    //      guarantees a result that is mathematically distinct from the others.
    // 2. The Purpose:
    //    - In real liquids, not every droplet moves at the same speed. Some are
    //      caught in faster currents, while others drag behind due to friction.
    //    - We use `hSpeed` to simulate this. One particle might complete its
    //      orbit in 2 seconds, while its neighbor takes 3 seconds.
    // 3. Avoiding Uniformity:
    //    - Without this variance, the entire liquid would rotate like a solid
    //      plastic disk. Adding random speeds makes the motion look fluid, chaotic,
    //      and alive.
    float hSpeed = hash(i);
    
    // PHYSICS PROGRESS
    
    // --- The Staggered Trigger (The "Rain" Effect) ---
    // Calculates exactly where *this specific particle* is in its lifecycle,
    // separate from the global pull progress.
    // 1. Breaking the Monolith:
    //    - If we just used `pull`, every single particle would start moving at
    //      the exact same time. The liquid would look like a solid elevator moving down.
    // 2. The Random Delay (hY * Offset):
    //    - We subtract `hY * FALL_STAGGER_OFFSET` from the global `pull`.
    //    - This means particles with a low `hY` start falling immediately, while
    //      particles with a high `hY` "wait" until the user has pulled further.
    // 3. The Local Time Scale (/ Scale):
    //    - We divide by `FALL_STAGGER_SCALE` to make the individual particle's
    //      transition faster than the total pull duration.
    //    - The result is a cascade: some drops fall early, some fall late, creating
    //      a lush, organic "raining" motion rather than a rigid slide.
    float rawProgress = saturate((pull - hY * FALL_STAGGER_OFFSET) / FALL_STAGGER_SCALE);
    // --- The Acceleration Variance (The "Personality" Curve) ---
    // Calculates a unique easing exponent for this specific particle.
    // 1. Defining the Spectrum:
    //    - We use `mix` to pick a value between `0.5` and `SPEED_VARIATION` based
    //      on the particle's random hash (`hSpeed`).
    // 2. The Physics of Exponents:
    //    - A value of `1.0` would mean linear movement (robotic constant speed).
    //    - A value < 1.0 (like 0.5) creates an "Ease-Out" (starts fast, slows down).
    //    - A value > 1.0 creates an "Ease-In" (starts slow, speeds up).
    // 3. The Visual Result:
    //    - By assigning a random exponent to every drop, we ensure they don't just
    //      start at different times (stagger), but they also *move* differently.
    //    - Some drops will "shoot" down quickly, while others will appear to
    //      "drag" or accelerate slowly, creating a chaotic, organic flow.
    float randomExponent = mix(0.5, SPEED_VARIATION, hSpeed);
    // --- The Motion Curve (The Non-Linear Easing) ---
    // Applies the specific acceleration "personality" to the particle's movement.
    // 1. The Power Function:
    //    - We take the linear progress (0.0 to 1.0) and raise it to the power of
    //      our random exponent.
    //    - If the exponent is 1.0, the motion is linear (robotic).
    //    - If the exponent is 2.0, the motion starts slow and speeds up (Ease-In).
    //    - If the exponent is 0.5, the motion starts fast and slows down (Ease-Out).
    // 2. The Result:
    //    - Since every particle has a unique `randomExponent`, no two particles
    //      accelerate exactly the same way.
    //    - This creates a chaotic, organic flow where some drops feel heavy and slow,
    //      while others feel light and fast, simulating real fluid dynamics.
    float individualProgress = pow(rawProgress, randomExponent);
    // --- The Final Polish (Smoothing the Motion) ---
    // Takes the unique "personality" curve we just created and softens the edges.
    // 1. The "S-Curve" Standard:
    //    - While the previous `pow()` function adjusted the speed (making it faster
    //      or slower), it can still result in abrupt starts or stops.
    //    - `smoothstep` forces the motion to start slowly (acceleration) and end
    //      slowly (deceleration), creating that classic "Ease-In-Ease-Out" feel.
    // 2. Combining Forces:
    //    - By chaining `pow` (personality) and `smoothstep` (softness), we get
    //      a "Weighted S-Curve."
    //    - This means every particle has a unique trajectory, but they all share
    //      the same fluid, viscous quality that makes them look like liquid
    //      rather than rigid geometric shapes.
    float easedProgress = smoothstep(0.0, 1.0, individualProgress);
    
    // STANDARD POSITIONS
    
    // --- The Shape Definition (The Funnel Geometry) ---
    // Calculates the horizontal boundary for the particle at its specific height.
    // 1. Defining the Cone:
    //    - We want the liquid to look like a funnel or a tornado: wide at the top
    //      (where it connects to the nav bar) and narrow at the bottom (the tip).
    // 2. The Mix Function (Linear Interpolation):
    //    - `uniforms.triangleWidths.y` represents the WIDTH at the BOTTOM (the tip).
    //    - `uniforms.triangleWidths.x` represents the WIDTH at the TOP (the base).
    // 3. The Interpolation Factor (hY):
    //    - We use the particle's random vertical position `hY` (0.0 to 1.0) as the mixer.
    //    - If `hY` is 0.0 (bottom), the width is `triangleWidths.y` (narrow).
    //    - If `hY` is 1.0 (top), the width is `triangleWidths.x` (wide).
    //    - This creates a perfect, mathematical triangle shape that confines the
    //      chaos of the particles into a recognizable stream.
    float triangleWidth = mix(parameters.triangleBottom, parameters.triangleTop, hY);
    // --- The Vertical Dispersion (The Starting Volume) ---
    // Calculates the initial Y-position for the particle based on its random "hY" seed.
    // 1. Defining the Ceiling (VERTICAL_SPREAD_START):
    //    - This constant represents the highest point in our particle cloud.
    //    - Think of it as the "top surface" of the liquid stored inside the notch
    //      or dynamic island before it is pulled down.
    // 2. Creating the Volume (hY * DEPTH):
    //    - We multiply the particle's unique `hY` (0.0 to 1.0) by the `DEPTH`.
    //    - This calculates a random vertical offset. If `hY` is 0.0, the offset is 0.
    //      If `hY` is 1.0, the offset is the full depth.
    // 3. The Subtraction Logic:
    //    - By subtracting the offset from the start, we scatter the particles
    //      downwards from the ceiling.
    //    - Instead of a single flat line of pixels, this creates a "thick" band
    //      of liquid, giving the illusion of volume and quantity to the fluid.
    float randomStartY = VERTICAL_SPREAD_START - (hY * VERTICAL_SPREAD_DEPTH);
    
    // --- The Starting Coordinate (The Funnel Mapping) ---
    // Calculates the precise 2D position (X, Y) where the particle begins its journey.
    // 1. Centering the Randomness (X-Axis):
    //    - `hX` is between 0.0 and 1.0. Subtracting 0.5 shifts the range to
    //      -0.5 to 0.5. This allows us to spread particles evenly to the left and
    //      right of the center line, rather than just extending to the right.
    // 2. Applying the Funnel Shape (* triangleWidth):
    //    - We multiply this centered value by `triangleWidth`.
    //    - Near the top (hY=1.0), the width is large, so particles spread wide.
    //    - Near the bottom (hY=0.0), the width is tiny, constraining particles
    //      tightly together. This creates the inverted triangle geometry.
    // 3. Screen Centering (+ screenSize.x / 2.0):
    //    - Finally, we add half the screen width. This takes our local coordinate
    //      (centered around 0) and moves it to the physical center of the device
    //      screen, ensuring the funnel is perfectly aligned.
    // 4. Vertical Position (Y-Axis):
    //    - We simply use the `randomStartY` calculated earlier to stagger the
    //      particles vertically within the top volume.
    float2 startPos = float2(
                             (hX - 0.5) * uniforms.screenSize.x * triangleWidth + (uniforms.screenSize.x / 2.0),
                             randomStartY
                             );
    
    // --- The Target Origin (The Ring's Center) ---
    // Calculates the absolute center point of the screen where the liquid will converge.
    // 1. Device Independence:
    //    - By dividing the `screenSize` (width, height) by 2.0, we dynamically
    //      locate the middle of the view regardless of the device (iPhone SE vs
    //      iPhone Pro Max).
    // 2. The Destination:
    //    - While `startPos` defined where the liquid *falls from* (the funnel),
    //      `circleCenter` defines where the liquid *lands*.
    //    - This point acts as the anchor for the loading circle. All particles
    //      will eventually rotate around this coordinate.
    // 3. Vector Math:
    //    - `screenSize` is a `float2`. In Metal, dividing a vector by a scalar (2.0)
    //      divides both X and Y components simultaneously, making this a highly
    //      efficient operation.
    float2 circleCenter = uniforms.screenSize / 2.0;
    // --- The Angular Distribution (The Slice Calculation) ---
    // Determines the precise angle (in radians) for this specific particle on the circle.
    // 1. Normalization (i / totalDots):
    //    - We take the particle's index `i` (e.g., 50) and divide it by the total count
    //      (e.g., 100). This gives us a percentage from 0.0 to 1.0.
    //    - Example: Particle 50/100 = 0.5 (Halfway through the list).
    // 2. Mapping to Radians (* M_PI_F * 2.0):
    //    - A full circle is 2 * Pi radians (approx 6.28).
    //    - By multiplying our percentage by this constant, we map 0.0-1.0 to 0-360 degrees.
    //    - Particle 0 gets 0 degrees. Particle 50 gets 180 degrees. Particle 100 gets 360 degrees.
    // 3. The Result:
    //    - This ensures that no matter how many particles we have, they are perfectly
    //      spaced out to form a complete, seamless ring.
    float angle = (i / totalDots) * M_PI_F * 2.0;
    // --- The Ring Thickness (The Radial Scatter) ---
    // Calculates a random offset from the perfect circle radius for this particle.
    // 1. Centering the Jitter (hY - 0.5):
    //    - `hY` is a random value between 0.0 and 1.0.
    //    - Subtracting 0.5 shifts the range to -0.5 to 0.5. This means some
    //      particles will be pushed *inside* the circle (-), and some will be
    //      pushed *outside* (+).
    // 2. Applying the Band Width (* PARTICLE_BAND_WIDTH):
    //    - We multiply this centered value by our constant (e.g., 20.0).
    //    - This defines the total thickness of the ring. If the width is 20,
    //      particles can be up to 10 pixels inside or 10 pixels outside the ideal path.
    // 3. The Visual Result:
    //    - Instead of a perfect, geometric line (which looks vector-based and fake),
    //      we get a "fuzzy" or "thick" band of particles. This simulates a real
    //      fluid that has surface tension and volume.
    float randomRadialOffset = (hY - 0.5) * PARTICLE_BAND_WIDTH;
    
    // --- The Final Radius (The Jittered Distance) ---
    // Calculates the unique distance from the center for this specific particle.
    // 1. The Ideal Circle (FORMATION_RADIUS):
    //    - This is the "perfect" radius (e.g., 50.0). If every particle used just
    //      this value, they would form a single-pixel-wide line.
    // 2. Adding the Variance (+ randomRadialOffset):
    //    - We add the offset calculated in the previous step (e.g., -5.0 or +3.0).
    //    - If the offset is negative, the particle sits slightly closer to the center.
    //    - If positive, it sits slightly further out.
    // 3. The Result:
    //    - Instead of a rigid geometry, the particles are scattered around the
    //      ideal path, creating a "thick" band that looks like a real, viscous fluid
    //      held together by surface tension.
    float baseRadius = FORMATION_RADIUS + randomRadialOffset;
    
    // --- The Elastic Overshoot (The "Jiggle" Physics) ---
    // Calculates a decaying sine wave to simulate surface tension and elasticity.
    // 1. The Oscillation (sin):
    //    - `sin(individualProgress * BOUNCE_FREQ)` creates a wave that cycles
    //      back and forth as the particle moves.
    //    - This simulates the liquid "wobbling" or fighting against air resistance.
    // 2. The Dampening (1.0 - easedProgress):
    //    - We multiply the wave by `(1.0 - easedProgress)`.
    //    - At the start (0.0), the bounce is strong. As the particle approaches
    //      the destination (1.0), this factor drops to 0.
    //    - This acts as a "friction" or "brake," ensuring the particle settles
    //      smoothly into place instead of vibrating forever at the target.
    // 3. The Visual Result:
    //    - This makes the liquid feel viscous and heavy. When the drops fall,
    //      they don't just slide linearly; they expand and contract slightly,
    //      giving the animation a playful, organic "bounciness."
    float radialBounce = sin(individualProgress * BOUNCE_FREQ) * BOUNCE_AMP * (1.0 - easedProgress);
    
    // --- The Polar-to-Cartesian Conversion (The Target Lock) ---
    // Converts our angle and radius (Polar) into X and Y coordinates (Cartesian).
    // 1. Determining Direction (Unit Vector):
    //    - `float2(cos(angle), sin(angle))` creates a vector of length 1.0 pointing
    //      in the exact direction of the particle's slice of the pie.
    // 2. Applying Distance (* Magnitude):
    //    - We multiply this direction by our calculated distance (`baseRadius + radialBounce`).
    //    - This stretches the vector out from the center to the exact spot where
    //      the particle should be, factoring in both the ring's size and the
    //      "jelly" wobble effect.
    // 3. Centering the System (+ circleCenter):
    //    - The math above assumes the center of the world is (0,0).
    //    - By adding `circleCenter` (screen width/2, screen height/2), we shift
    //      the entire calculated shape to the middle of the iPhone screen.
    float2 targetPosWithBounce = circleCenter + float2(cos(angle), sin(angle)) * (baseRadius + radialBounce);
    
    // CALCULATE BASE PATH
    
    // --- The Morphing Calculation (The Linear Path) ---
    // Mathematically interpolates the particle's position between its origin
    // and its destination.
    // 1. The "Mix" Function (Lerp):
    //    - This is a hardware-optimized Linear Interpolation function.
    //    - It takes two points (A and B) and a percentage (0.0 to 1.0).
    //    - If `easedProgress` is 0.0, the result is `startPos` (the funnel).
    //    - If `easedProgress` is 1.0, the result is `targetPos` (the ring).
    // 2. The Trajectory:
    //    - By smoothly changing `easedProgress` from 0 to 1 over time, the
    //      particle slides perfectly along a straight line connecting the
    //      waterfall to the loading circle.
    // 3. Efficiency:
    //    - Doing this calculation on the GPU allows us to animate thousands of
    //      particles simultaneously without any CPU overhead.
    float2 basePos = mix(startPos, targetPosWithBounce, easedProgress);
    
    // TANGENTIAL WIGGLE (Snaking Effect)
    
    // --- The Orientation Vector (The "Compass") ---
    // Calculates the normalized direction from the start point to the end point.
    // 1. Vector Subtraction (B - A):
    //    - `targetPosWithBounce - startPos` gives us the full vector connecting
    //      the two points. It contains both direction and distance.
    // 2. Normalization (Making it a Unit Vector):
    //    - The `normalize()` function divides the vector by its own length.
    //    - The result is a vector with a length of exactly 1.0.
    //    - This discards the "distance" info and leaves us with pure "direction."
    // 3. Why we need this:
    //    - Real liquid droplets deform as they move fast. They stretch out.
    //    - We will use `travelDir` later to rotate and stretch the particle
    //      so that the "tail" of the droplet trails behind its movement,
    //      creating a realistic motion blur effect.
    float2 travelDir = normalize(targetPosWithBounce - startPos);
    // --- The Cross Vector (The "Ribs") ---
    // Calculates a vector that is perfectly perpendicular (90 degrees) to the
    // direction of travel.
    // 1. The Math Trick (Rotation):
    //    - To rotate a 2D vector 90 degrees, you swap the X and Y components and
    //      negate one of them.
    //    - If `travelDir` is (1, 0) [Right], `sideDir` becomes (0, 1) [Up].
    // 2. Why we need this:
    //    - `travelDir` controls the "Spine" (length/stretch) of the droplet.
    //    - `sideDir` controls the "Ribs" (width/thickness) of the droplet.
    // 3. The Result:
    //    - This gives us a local coordinate system for the particle. We can now
    //      stretch it along the travel path to make it look fast (motion blur)
    //      while keeping it narrow along the side path.
    float2 sideDir = float2(-travelDir.y, travelDir.x); // Perpendicular vector
    
    // --- The Wiggle Trigger (The Timing Control) ---
    // Defines a "window of opportunity" for the particle to wiggle sideways.
    // 1. The Startup (smoothstep 0.2 -> 0.8):
    //    - We don't want the particle to wiggle instantly when it spawns at the top
    //      (0.0). It needs to pick up speed first.
    //    - The `smoothstep` ensures the effect ramps up gently as the particle falls.
    // 2. The Cutoff (1.0 - easedProgress):
    //    - We also don't want the particle to still be wiggling when it hits the
    //      target ring (1.0). It needs to land cleanly.
    //    - This multiplier forces the wiggle to fade out completely as the particle
    //      approaches its destination.
    // 3. The Result:
    //    - The wiggle only happens in the "middle" of the fall—simulating air
    //      turbulence or instability during high-speed travel—while keeping the
    //      start and end points stable.
    float wiggleEnvelope = smoothstep(0.2, 0.8, individualProgress) * (1.0 - easedProgress);
    // --- The Wiggle Frequency (The Vibration Speed) ---
    // Determines how fast this specific particle oscillates back and forth.
    // 1. The Base Speed (BASE_WIGGLE_FREQ):
    //    - This provides the "standard" rhythm for the liquid stream, ensuring
    //      the motion feels consistent with the overall fluid simulation.
    // 2. Breaking the Synchronization (+ i * WIGGLE_VARIATION):
    //    - By adding a tiny bit of the particle's ID (`i`) to the frequency,
    //      every drop gets a slightly different "heartbeat."
    //    - Particle #1 might wiggle 12 times per second, while Particle #50
    //      wiggles 17 times per second.
    // 3. The Visual Result:
    //    - This prevents the "marching band" effect where all particles move
    //      as a single rigid unit. Instead, they look like independent droplets
    //      drifting through a turbulent air current.
    float wiggleFreq = BASE_WIGGLE_FREQ + (i * WIGGLE_VARIATION);
    
    // --- The Side-to-Side Motion (The "Snake" Offset) ---
    // Calculates the physical displacement for the particle's "wiggle" effect.
    // 1. Determining Direction (sideDir *):
    //    - We multiply the entire calculation by `sideDir`. This ensures that the
    //      wiggle only pushes the particle left or right relative to its path,
    //      never speeding it up or slowing it down along its travel line.
    // 2. The Oscillation (sin):
    //    - We feed the `individualProgress` (time) and `wiggleFreq` into a sine wave.
    //    - This creates the back-and-forth movement that makes the drop look like
    //      it's falling through wind or turbulence.
    // 3. The Envelope Control (* WIGGLE_AMP * wiggleEnvelope):
    //    - `WIGGLE_AMP` sets the maximum distance of the drift.
    //    - `wiggleEnvelope` acts as a "volume knob," turning the wiggle off at
    //      the start and end of the journey so the drop lands perfectly on target.
    float2 wiggleOffset = sideDir * sin(individualProgress * wiggleFreq) * WIGGLE_AMP * wiggleEnvelope;
    
    // --- The Final Path Assembly (Adding Chaos to Logic) ---
    // Combines the calculated linear trajectory with the random wiggle offset.
    // 1. Vector Addition:
    //    - By using `+=`, we take the particle's "ideal" position (basePos) and
    //      nudge it by the "turbulent" position (wiggleOffset).
    // 2. The Physical Result:
    //    - Instead of moving in a boring, straight line from the notch to the ring,
    //      the particle now follows a "snaking" or "drifting" path.
    // 3. Layered Motion:
    //    - Because `wiggleOffset` is perpendicular to the travel direction, this
    //      addition doesn't change *how far* the particle has fallen; it only
    //      changes *how wide* it drifts during the fall.
    // 4. State Update:
    //    - `basePos` now represents the true, final coordinate of the particle center
    //      for this specific frame of animation.
    basePos += wiggleOffset;
    
    // TURBULENCE (Global Drift)
    
    // --- The Transition Switch (The Landing Logic) ---
    // A value that tracks how much of the "falling" state remains.
    // 1. Inverse Mapping:
    //    - Since `easedProgress` goes from 0.0 (top) to 1.0 (bottom),
    //      `isFalling` does the opposite: it starts at 1.0 and shrinks to 0.0.
    // 2. The Multiplier Effect:
    //    - We use this as a weight. Any effect multiplied by `isFalling`
    //      will be at full strength while the particle is in the air,
    //      but will "shut off" the moment it hits the target ring.
    // 3. Visual Stability:
    //    - This ensures that messy "airborne" physics (like motion blur or
    //      high-speed stretching) don't jitter the final loading circle,
    //      allowing the particles to settle into a clean, static shape.
    float isFalling = (1.0 - easedProgress);
    // --- The Stability Control (The Chaos Clutch) ---
    // Gradually reduces the intensity of erratic movement as the particle lands.
    // 1. Morph Phase Relationship:
    //    - `morphPhase` represents how far the particle has transformed into its
    //      final state. By subtracting it from 1.0, we create an inverse relationship.
    // 2. The Dampening Effect:
    //    - When the drop is high in the air (morphPhase near 0), the dampener is 1.0
    //      (Full Chaos).
    //    - As it merges into the circle (morphPhase near 1), the dampener hits 0.0
    //      (Zero Chaos).
    // 3. Why we use it:
    //    - This prevents the "jitter" or "popping" that would happen if the
    //      turbulent forces (like wind or wiggle) suddenly stopped. It ensures
    //      the transition from "messy liquid" to "perfect geometric circle"
    //      is silky smooth and visually satisfying.
    float turbulenceDampener = (1.0 - morphPhase);
    // --- The Micro-Turbulence (The High-Frequency Shiver) ---
    // Calculates a fast, time-based horizontal jitter to simulate air resistance.
    // 1. Time-Driven Oscillation (sin):
    //    - By multiplying `uniforms.time` by `TURB_FREQ_X`, we create a very fast
    //      vibration that never stops.
    //    - Adding `i` ensures that every particle is at a different point in its
    //      shiver cycle, so they don't all vibrate in unison.
    // 2. The Power of the Dampeners:
    //    - We multiply by both `isFalling` and `turbulenceDampener`.
    //    - This ensures the shivering is violent mid-air but completely disappears
    //      the moment the drop reaches the loading ring.
    // 3. The Visual Result:
    //    - This prevents the drops from looking like "clean" computer-generated
    //      circles. It adds a layer of "organic noise" that makes the liquid
    //      feel like it's struggling against the wind as it falls.
    float turbX = sin(uniforms.time * TURB_FREQ_X + i) * TURB_AMP_X * isFalling * turbulenceDampener;
    // --- The Vertical Jitter (The Shiver Y-Axis) ---
    // Calculates a fast, time-based vertical vibration to complement the X-axis jitter.
    // 1. The Phase Shift (cos):
    //    - By using `cos` here while the X-axis used `sin`, we create a circular
    //      displacement. If both used `sin`, the particle would only move diagonally.
    //    - This "offset" phase makes the droplet appear to tumble or rotate slightly
    //      in place as it falls.
    // 2. Frequency & Amplitude:
    //    - `TURB_FREQ_Y` and `TURB_AMP_Y` allow us to tune the vertical "shake"
    //      independently from the horizontal one, as air resistance usually
    //      affects drops differently along their line of travel.
    // 3. Controlled Fade-Out:
    //    - Just like `turbX`, this is multiplied by the dampeners. This ensures
    //      the "noise" is stripped away as the particle settles, leaving a
    //      perfectly still and clean loading ring.
    float turbY = cos(uniforms.time * TURB_FREQ_Y + i) * TURB_AMP_Y * isFalling * turbulenceDampener;
    // --- The Final Global Position (The Memory Commit) ---
    // Writes the final, fully-processed coordinate into the particle's memory.
    // 1. Combining Macro and Micro Motion:
    //    - `basePos` already contains the journey (start to end) and the large
    //      "snaking" wiggle.
    //    - By adding `float2(turbX, turbY)`, we layer on the high-frequency
    //      shiver (micro-turbulence).
    // 2. Buffer Storage:
    //    - `particles[id].position` is a pointer to the actual GPU memory
    //      representing this specific particle.
    //    - Once this line executes, the "Vertex Shader" (the next stage of
    //      the pipeline) will use this exact coordinate to draw the pixel on the screen.
    // 3. The Visual "Soul":
    //    - Because this is calculated every single frame (usually 60 or 120
    //      times per second), these small additions create the illusion of
    //      life, making a static mathematical formula look like a living,
    //      breathing fluid.
    particles[id].position = basePos + float2(turbX, turbY);
    
    // --- ALPHA & SIZE (Unchanged) ---
    
    // --- The Proximity Sensor (The Arrival Logic) ---
    // A normalized value that detects if the particle has reached its final destination.
    // 1. Defining the Threshold (CIRCLE_DETECT_LIMIT):
    //    - We don't wait for the particle to be exactly at 1.0 (100% finished).
    //    - By starting the transition at a limit (e.g., 0.8), we allow for a
    //      gradual "fade-in" of the final circle properties.
    // 2. The Smoothstep Transition:
    //    - `smoothstep(0.8, 1.0, easedProgress)` returns 0.0 while the drop is
    //      falling, and ramps up to 1.0 only as it enters the home stretch.
    // 3. Why we need this:
    //    - This is used to "swap" behaviors. For example, we can use this to
    //      turn off the droplet's elongated "tail" and turn on its perfect
    //      spherical shape the moment it joins the loading ring.
    //    - It ensures the liquid "snaps" into the geometric ring with
    //      clean, crisp precision.
    float isInsideCircle = smoothstep(CIRCLE_DETECT_LIMIT, 1.0, easedProgress);
    // --- The Opacity Transition (The Solidification) ---
    // Smoothly blends the particle's transparency between its "airborne"
    // and "docked" states.
    // 1. Defining the Two States:
    //    - `OPACITY_FALLING`: Usually a lower value (e.g., 0.6). This makes the
    //      falling stream look light, fast, and slightly motion-blurred.
    //    - `OPACITY_IN_CIRCLE`: Usually a higher value (e.g., 1.0). This makes
    //      the final loading ring look solid, stable, and easy to read.
    // 2. The Weight (isInsideCircle):
    //    - We use the proximity sensor we just calculated as the "mixer."
    //    - As the particle enters the circle boundary, it gradually becomes
    //      more opaque.
    // 3. The Visual Polish:
    //    - This prevents the loading ring from looking cluttered. By keeping
    //      the falling "tails" slightly transparent, the user's eye is
    //      naturally drawn to the more solid, completed part of the animation.
    float currentAlpha = mix(OPACITY_FALLING, parameters.opacityInCircle, isInsideCircle);
    // --- The Final Transparency (The Morph Completion) ---
    // Calculates the ultimate visibility of the particle based on the global morph state.
    // 1. The Starting Point (currentAlpha):
    //    - This already accounts for whether the particle is falling (semi-transparent)
    //      or has reached the circle (more solid).
    // 2. The Morph Influence (morphPhase):
    //    - As the overall system transitions from "The Waterfall" into "The Final Ring,"
    //      `morphPhase` moves from 0.0 to 1.0.
    //    - We use this to force the particles toward 1.0 (100% opaque).
    // 3. The Visual Logic:
    //    - While individual drops might have varying opacities to look "liquid,"
    //      the final UI element (the loading circle) needs to be perfectly solid
    //      and clear. This line ensures that no matter how transparent a drop was
    //      while falling, it becomes a "real" UI pixel once the animation completes.
    float finalAlpha = mix(currentAlpha, 1.0, morphPhase);
    
    // --- The Handoff Logic (The Ghosting Effect) ---
    // Gradually fades the individual particles out as the solid ring texture fades in.
    // 1. The Inverse Relationship (1.0 - solidRingAlpha):
    //    - `solidRingAlpha` represents the visibility of the "perfect" geometric ring.
    //    - By subtracting it from 1.0, we create a "mask." When the solid ring is
    //      invisible (0.0), this value is 1.0. When the solid ring is fully
    //      visible (1.0), this value is 0.0.
    // 2. The Multiplication (*=):
    //    - We multiply the particle's alpha by this mask. This ensures that the
    //      individual drops "give up" their visibility as the solid UI element
    //      takes over.
    // 3. Why we do this:
    //    - If we didn't fade the particles out, the final ring would look "lumpy"
    //      because you would see the individual dots stacked on top of the solid
    //      line. This ensures a silky-smooth transition from a "fluid simulation"
    //      to a "UI component."
    finalAlpha *= (1.0 - solidRingAlpha);
    // --- The Presence Control (The Gesture Tether) ---
    // Ties the visibility of the entire particle system directly to the pull distance.
    // 1. The "Pull" Multiplier:
    //    - `pull` is a value (0.0 to 1.0) representing how far the user has
    //      dragged the screen down.
    //    - By multiplying the total alpha by this value, we ensure that if the
    //      user hasn't pulled yet (0.0), the particles are 100% invisible.
    // 2. The Dynamic Entry:
    //    - As the user pulls, the liquid doesn't just "pop" into existence.
    //      It "bleeds" onto the screen, starting as a faint ghost and becoming
    //      solidly visible only as the tension increases.
    // 3. Cleanup:
    //    - This also handles the "snap back" effect. If the user lets go without
    //      triggering the refresh, `pull` drops back to 0, causing all the
    //      particles to instantly—but smoothly—vanish.
    finalAlpha *= pull;
    
    // --- The Final Shading (The RGBA Assembly) ---
    // Combines the RGB color channels with the calculated transparency (Alpha).
    // 1. Theme Consistency (uniforms.color.rgb):
    //    - We take the Red, Green, and Blue values directly from the app's
    //      theme settings. This ensures the particles perfectly match the
    //      branding of the rest of the UI.
    // 2. The 4-Component Vector (float4):
    //    - Metal requires colors to be in a 4-part format: (Red, Green, Blue, Alpha).
    //    - We "pack" our fixed RGB values together with our highly dynamic
    //      `finalAlpha` (which contains the pull logic, the morphing, and the
    //      handoff logic).
    // 3. The GPU Output:
    //    - `particles[id].color` sends this information to the hardware's
    //      rasterizer.
    //    - This is the final step for this particle in the compute kernel;
    //      it is now fully positioned, shaped, and colored, ready to be
    //      drawn on the user's screen.
    particles[id].color = float4(parameters.color.rgb, finalAlpha);
    
    // --- The Size Tiering (The Population Diversity) ---
    // Categorizes particles into three distinct size groups to create a more
    // natural, non-uniform liquid look.
    // 1. Why use Tiers instead of pure Random?
    //    - If every particle was a completely random size, the liquid would look
    //      like "static" or "noise."
    //    - By creating three specific sizes (Large, Medium, Small), we simulate
    //      droplets of different volumes, which looks much more like real fluid.
    // 2. The Logic (Ternary Branching):
    //    - We use the random seed `hS` (0.0 to 1.0) to assign a category:
    //    - 40% of particles become LARGE (1.0).
    //    - 40% of particles become MEDIUM (0.8).
    //    - 20% of particles become SMALL (0.5).
    // 3. The Visual Result:
    //    - This variety adds "texture" to the stream. The larger drops act as
    //      the "core" of the liquid, while the smaller ones act as the "spray"
    //      or "mist" around the edges.
    float randomSizeFactor = (hS < SIZE_TIER_LARGE_THRESHOLD) ? SIZE_SCALE_LARGE :
    ((hS < SIZE_TIER_MEDIUM_THRESHOLD) ? SIZE_SCALE_MEDIUM : SIZE_SCALE_SMALL);
    
    // --- The Base Dimension (The Physical Radius) ---
    // Calculates the starting pixel size for the particle before any
    // animation-based scaling is applied.
    // 1. The Global Base (DOT_SIZE_FALLING_BASE):
    //    - This is the "standard" size for your droplets (e.g., 4.0 pixels).
    //    - It acts as the master control for the thickness of the falling stream.
    // 2. Applying the Tier (* / randomSizeFactor):
    //    - We use the `randomSizeFactor` calculated in the previous step to
    //      modify the base size.
    //    - Because we are dividing, the "Small" tier (0.5) will actually create
    //      the largest droplets, while the "Large" tier (1.0) keeps them at
    //      the base size. This inversion helps ensure that smaller counts of
    //      large drops feel as impactful as large counts of small drops.
    // 3. The Variance (* DOT_SIZE_VARIANCE):
    //    - A final multiplier (e.g., 1.2) that allows you to globally "inflate"
    //      or "deflate" the entire particle system without changing individual
    //      tier logic.
    float pSize = (DOT_SIZE_FALLING_BASE / randomSizeFactor) * DOT_SIZE_VARIANCE;
    
    // --- The Size Morph (The Geometric Transition) ---
    // Smoothly transitions the particle's diameter from its "liquid" state
    // to its "UI element" state.
    // 1. The Source (pSize):
    //    - This is the "random" size the particle had while falling. Some are
    //      large, some are small, creating the organic look of splashing water.
    // 2. The Destination (DOT_SIZE_TARGET):
    //    - This is a fixed constant. Every dot in the final loading circle
    //      must be exactly the same size to look professional and intentional.
    // 3. The Interpolation (mix ... morphPhase):
    //    - As `morphPhase` goes from 0.0 to 1.0, the GPU performs a weighted
    //      average between the random falling size and the perfect target size.
    // 4. The Result:
    //    - Instead of "popping" from one size to another, the droplets
    //      gracefully inflate or deflate as they land, ensuring the final
    //      ring looks perfectly uniform without any jitter.
    particles[id].size = mix(pSize, DOT_SIZE_TARGET, morphPhase);
}


// --- The Vertex Shader (The Sculptor) ---
// This function runs for every single vertex of every particle (6 vertices per square).
// Its job is to map the particle's physical data to the actual screen coordinates.
// 1. [[vertex_id]]:
//    - A unique index for the current vertex being processed.
//    - If we are drawing squares (quads), this will count 0, 1, 2, 3, 4, 5
//      for the first particle, 6-11 for the second, and so on.
// 2. [[buffer(0)]]:
//    - This is the "Result Buffer" from your Compute Kernel.
//    - It contains the final position, color, and size for every particle
//      that we just calculated.
// 3. [[buffer(1)]]:
//    - Global data (screen size, current time) shared by all particles.
//    - We use this to ensure the particles are scaled correctly relative
//      to the iPhone's screen resolution.
vertex VertexOut particleVertex(
                                uint vertexID [[vertex_id]],
                                const device Particle *particles [[buffer(0)]],
                                constant Uniforms &uniforms [[buffer(1)]])
{
    // --- The Output Container (The Fragment Messenger) ---
    // Creates an instance of the struct that will be passed from this
    // Vertex Stage to the Fragment (Pixel) Stage.
    // 1. Data Bridging:
    //    - The Vertex Shader calculates where things are on the screen.
    //    - The Fragment Shader calculates what color each individual pixel should be.
    //    - `out` acts as the bridge between these two worlds.
    // 2. Interpolation:
    //    - Any data we put into `out` (like color or texture coordinates) will be
    //      automatically "blended" across the surface of the particle by the GPU
    //      before it reaches the pixel-drawing stage.
    // 3. Efficiency:
    //    - By initializing this here, we provide a clean workspace to fill in
    //      the position and color data that the hardware's rasterizer requires.
    VertexOut out;
    // --- The Data Lookup (The Blueprint Retrieval) ---
    // Uses the current thread's ID to fetch the specific simulation data
    // required to draw this specific part of the particle system.
    // 1. The Index (vertexID):
    //    - This is the "ticket number" assigned to the current vertex.
    //    - It tells the shader exactly which slot in the array corresponds
    //      to the geometry we are about to process.
    // 2. The Fetch (particles[...]):
    //    - We reach into the shared GPU memory buffer to grab the struct
    //      containing the physics results (position) and visual traits (color).
    // 3. The Local Copy (Particle p):
    //    - We unpack this data into a local variable 'p'.
    //    - This acts as a handy shortcut, letting us access properties like
    //      'p.position' cleanly without repeatedly querying the global buffer.
    Particle p = particles[vertexID];
    // --- The Screen-to-GPU Projection (The Mapping) ---
    // Converts pixel coordinates (0 to ScreenWidth) into GPU coordinates (-1 to 1).
    // 1. Normalization (position / screenSize):
    //    - This converts the position into a percentage of the screen (0.0 to 1.0).
    //    - If a particle is in the middle of a 400px screen, 200/400 = 0.5.
    // 2. Range Expansion (* 2.0):
    //    - The GPU's visible window is 2 units wide (from -1 on the left to 1 on the right).
    //    - We multiply our 0.0-1.0 range by 2.0 to get a 0.0-2.0 range.
    // 3. The Re-centering (- 1.0):
    //    - Finally, we shift everything left by 1 unit.
    //    - 0.0 (Left) becomes -1.0.
    //    - 1.0 (Middle) becomes 0.0.
    //    - 2.0 (Right) becomes 1.0.
    // 4. Why we do this:
    //    - Metal doesn't care if you have an iPhone 16 or an iPad; it always
    //      expects the screen to be a grid from -1 to +1. This math ensures
    //      your animation scales perfectly on any device.
    float2 ndc = (p.position / uniforms.screenSize) * 2.0 - 1.0;
    // --- The Vertical Correction (The Gravity Flip) ---
    // Reverses the Y-axis to align the GPU's coordinate system with the App's UI.
    // 1. The Conflict:
    //    - In standard iOS/SwiftUI design, (0, 0) is the TOP-left. Increasing Y
    //      moves you DOWN the screen.
    //    - In Metal (NDC), (0, 0) is the CENTER. Increasing Y moves you UP
    //      toward the top of the screen.
    // 2. The Solution (*= -1.0):
    //    - By negating the Y value, we flip the entire world upside down.
    //    - This aligns the math we did in the Compute Kernel (where we added
    //      gravity to move "down") with the way the GPU actually draws.
    // 3. The Result:
    //    - Your particles will now correctly originate from the "notch" area
    //      and fall toward the loading ring at the bottom, matching the
    //      user's physical "pull" gesture.
    ndc.y *= -1.0;
    // --- The Projection Output (The Homogeneous Coordinate) ---
    // Sets the final position of the vertex in Clip Space.
    // 1. 2D to 4D (float4):
    //    - Even though we are making a 2D animation, Metal's rendering pipeline
    //      operates in 4D space (X, Y, Z, W).
    // 2. Depth Control (0.0):
    //    - We set the Z-component to 0.0 because our particles don't need to
    //      move "into" or "out of" the screen. This keeps them on the
    //      primary drawing plane.
    // 3. The W-Component (1.0):
    //    - The value 1.0 is the "W" (homogeneous) coordinate. In 2D rendering,
    //      setting this to 1.0 ensures that our NDC coordinates (-1 to 1)
    //      are used exactly as they are without being scaled by perspective.
    // 4. The Result:
    //    - The GPU now knows the exact boundaries of where this vertex
    //      sits relative to the screen edges. Anything outside this range
    //      is automatically "clipped" (discarded) for performance.
    out.position = float4(ndc, 0.0, 1.0);
    out.color = p.color;
    out.size = p.size;
    // --- The Render Mode Toggle (The Identity Switch) ---
    // Categorizes the particle so the Pixel Shader knows which "look" to apply.
    // 1. Threshold Detection (p.size > 200.0):
    //    - We use a specific size threshold to distinguish between types of geometry.
    //    - Smaller values represent the individual "Waterfall" droplets.
    //    - Larger values (anything over 200.0) represent the "Main Ring" that
    //      appears once the liquid has finished falling.
    // 2. The Mode Flag (1.0 vs 0.0):
    //    - Mode 0.0 (Droplet): The Fragment Shader will draw a soft, gooey
    //      circle with organic edges.
    //    - Mode 1.0 (Solid Ring): The Fragment Shader will draw a clean,
    //      anti-aliased geometric arc.
    // 3. Why we use a float:
    //    - We pass this as a float (rather than a boolean) because GPU hardware
    //      is optimized for floating-point math. This allows the value to be
    //      interpolated or used directly in mixing functions without type casting.
    out.mode = (p.size > 200.0) ? 1.0 : 0.0;
    return out;
}
//--- The Fragment Shader (The Liquid Finisher) ---
// This function runs for every single pixel covered by your particles.
// It is responsible for sculpting the raw geometry into a smooth fluid.
// 1. [[stage_in]]:
//    - The "handover package" from the Vertex Shader.
//    - It contains interpolated data (like color or particle age) passed
//      down the pipeline.
// 2. [[point_coord]]:
//    - The "Internal Compass" (0.0 to 1.0) for this specific point primitive.
//    - Since particles are technically drawn as squares, we use this coordinate
//      system to calculate the distance from the center (0.5, 0.5).
//    - This allows us to discard the corners (making it round) and apply
//      radial gradients for that "soft liquid" look.
// 3. [[buffer(1)]]:
//    - Global scene state (like View/Projection matrices or Time).
// 4. [[buffer(2)]]:
//    - The "Tweakable Knobs."
//    - Contains specific fluid settings like base color, glow strength,
//      and smoothness thresholds to control the viscosity visualization.
fragment float4 particleFragment(
                                 VertexOut in [[stage_in]],
                                 float2 pointCoord [[point_coord]],
                                 constant Uniforms &uniforms [[buffer(1)]],
                                 constant Parameters &parameters [[buffer(2)]])
{
    // --- The Radial Coordinate (The Shape Foundation) ---
    // Calculates how far the current pixel is from the center of the particle.
    // 1. Centering (pointCoord - 0.5):
    //    - `pointCoord` runs from 0.0 to 1.0 across the square.
    //    - By subtracting 0.5, the center of the square becomes (0,0).
    // 2. The Distance Formula (length):
    //    - Calculates the "hypotenuse" from the center to the current pixel.
    //    - The center is 0.0, and the corners are roughly 0.707.
    // 3. Normalization (* 2.0):
    //    - We multiply by 2.0 so that the distance to the edge of the
    //      square is exactly 1.0.
    // 4. The Result:
    //    - This creates a "Signed Distance Field" (SDF).
    //    - Pixels at the center have a value of 0.0.
    //    - Pixels at the edges have a value of 1.0.
    //    - This value is the foundation for drawing a perfect circle.
    float dist = length(pointCoord - 0.5) * 2.0;
    // --- The Geometry Clip (The Performance Cut) ---
    // Instantly stops the processing of pixels that are outside the particle's radius.
    // 1. The Boundary (dist > 1.0):
    //    - Since we normalized our distance to 1.0 at the edges, any value
    //      greater than 1.0 is technically "outside" the circle we want to draw.
    // 2. The Discard Command:
    //    - `discard_fragment()` is a powerful instruction. It tells the GPU
    //      to throw away this pixel entirely.
    //    - It won't be colored, it won't be blended, and it won't be written
    //      to the screen buffer.
    // 3. Perfect Transparency:
    //    - This is how we turn a square "box" (the particle's primitive shape)
    //      into a circle. By discarding the corners of the square, we are
    //      left with a perfectly sharp circular boundary.
    if (dist > 1.0) discard_fragment();
    
    // --- The Alpha Canvas (The Shape Foundation) ---
    // Initializes the local transparency value for the pixel.
    // 1. Resetting the State:
    //    - We start at 0.0 (completely transparent).
    //    - This ensures that every pixel begins as "empty air" until we
    //      calculate exactly how much "liquid" (color) should occupy it.
    // 2. The Multi-Pass Target:
    //    - Since our shader has two modes (individual droplets vs. the solid ring),
    //      this variable acts as a temporary container that will be filled
    //      by different mathematical formulas depending on the 'in.mode' flag.
    // 3. Precision:
    //    - By using a float here, we can calculate ultra-smooth gradients,
    //      which is what gives the "pull-to-refresh" that high-end,
    //      fluid-like aesthetic.
    float alphaShape = 0.0;
    
    // --- The Style Branch (The Mode Selector) ---
    // Diverts the rendering path based on the particle's "Identity Flag."
    // 1. The Threshold (> 0.5):
    //    - Since 'mode' is a float, we check if it's greater than 0.5 to
    //      determine if this particle is in "Ring Mode."
    // 2. The Ring Logic (Mode 1.0):
    //    - If this condition is true, we are no longer drawing a soft,
    //      fuzzy droplet. We are now drawing a part of the solid,
    //      circular loading indicator.
    // 3. Mathematical Separation:
    //    - By branching here, we keep the "liquid math" (which needs softness)
    //      completely separate from the "ring math" (which needs crisp
    //      anti-aliasing), ensuring maximum performance and visual clarity.
    if (in.mode > 0.5) {
        // --- The Radial Normalization (The Scale Sync) ---
        // Converts the ring's physical radius into a coordinate-friendly ratio.
        // 1. Physical to Relative:
        //    - `FORMATION_RADIUS` is the actual size in points (e.g., 40pt).
        //    - `SOLID_CANVAS_SIZE * 0.5` is the distance from the center to
        //      the edge of the drawing area.
        // 2. The Ratio (uvRadius):
        //    - By dividing them, we get a value (usually around 0.8) that
        //      represents where the ring sits relative to the canvas edges.
        // 3. Why we do this:
        //    - This ensures the ring is drawn at the exact same size regardless
        //      of the device's screen resolution or the canvas size. It keeps
        //      the "Landing Zone" and the "Solid Ring" perfectly aligned.
        float uvRadius = FORMATION_RADIUS / (SOLID_CANVAS_SIZE * 0.5);
        // --- The Ring Offset (The Hollow Core Logic) ---
        // Measures how far the current pixel is from the "thin wire" of the ring.
        // 1. Center-Distance (dist):
        //    - We already know how far this pixel is from the center (0.0 at center, 1.0 at edge).
        // 2. The Target Radius (uvRadius):
        //    - This is the exact "sweet spot" where the line of the ring should sit.
        // 3. The Absolute Difference (abs):
        //    - By taking the absolute value of the difference, we treat the ring like a line.
        //    - If `dist` is 0.8 and `uvRadius` is 0.8, the result is 0.0 (you are on the line).
        //    - As you move inside OR outside that 0.8 mark, the value increases.
        // 4. Result:
        //    - This creates a "V-shaped" gradient where the "valley" (0.0) is the
        //      exact center-path of the loading ring.
        float distToLine = abs(dist - uvRadius);
        // --- The Stroke Normalization (The Thickness Sync) ---
        // Converts the physical dot size into a coordinate-friendly width.
        // 1. Matching the Droplets (DOT_SIZE_TARGET * 0.5):
        //    - We take half the diameter of the landing dots to get their radius.
        //    - This ensures that when the "morph" completes, the ring's stroke
        //      is exactly as wide as the dots that formed it.
        // 2. Coordinate Mapping (/ (SOLID_CANVAS_SIZE * 0.5)):
        //    - Just like the radius, we must convert this pixel value into
        //      the 0.0-1.0 range of our drawing canvas.
        // 3. Visual Consistency:
        //    - This is the "secret sauce" for a seamless transition. By
        //      calculating thickness this way, the solid ring won't appear
        //      too skinny or too chunky compared to the liquid droplets;
        //      it will look like the droplets have literally melted together.
        float uvThickness = (DOT_SIZE_TARGET * 0.5) / (SOLID_CANVAS_SIZE * 0.5);
        // --- The Optical Adjustment (The Core Thinning) ---
        // Refines the thickness of the ring to ensure visual balance.
        // 1. Defining the Core (uvThickness):
        //    - This is the "theoretical" thickness that matches the droplet size.
        // 2. The Thinning Factor (* RING_CORE_THINNING):
        //    - Geometric lines often look "heavier" to the human eye than
        //      individual dots of the same width.
        //    - By multiplying by a constant (e.g., 0.9), we slightly slim down
        //      the solid ring's core.
        // 3. The Result:
        //    - This prevents the transition from looking "bloated." It creates
        //      the illusion that the liquid has reached a state of surface
        //      tension, tightening up into a perfect, sleek geometric path.
        float coreThickness = uvThickness * RING_CORE_THINNING;
        // --- The Clean Stroke (The Anti-Aliased Edge) ---
        // Generates the solid alpha for the ring using a reversed smoothstep.
        // 1. The Distance Logic (distToLine):
        //    - Remember, `distToLine` is 0.0 at the center of the ring's path.
        //    - As we move away from the path, the value increases.
        // 2. The Inversion (High to Low):
        //    - By putting `coreThickness` as the first parameter and a smaller
        //      value second, the function returns 1.0 (fully opaque) inside
        //      the thickness and 0.0 (transparent) outside.
        // 3. Anti-Aliasing (- ANTIALIAS_SOFTNESS):
        //    - Instead of a hard "on/off" switch, we fade the alpha over a
        //      tiny fraction of a pixel.
        //    - This creates "sub-pixel" smoothness, making the ring look like
        //      a high-resolution vector graphic rather than a low-res bitmap.
        float core = smoothstep(coreThickness, coreThickness - ANTIALIAS_SOFTNESS, distToLine);
        // --- The Light Bleed (The Atmospheric Glow) ---
        // Creates a soft, secondary halo around the solid ring core.
        // 1. Inverting the Distance (1.0 - distToLine):
        //    - Since `distToLine` is 0.0 on the ring path, this gives us
        //      a value of 1.0 (max brightness) exactly on the line.
        // 2. The Spread (* RING_GLOW_SPREAD):
        //    - A constant that controls how "fat" the glow is. Increasing
        //      this makes the glow disappear faster as you move away from
        //      the center path.
        // 3. The Exponential Falloff (pow ... GLOW_FALLOFF_POWER):
        //    - By raising the value to a power (e.g., 2.0 or 3.0), we make
        //      the light fade out naturally. This mimics how real light
        //      scatters through a lens or fluid.
        // 4. Visual Cohesion:
        //    - This subtle glow masks the "perfect" edge of the geometric ring,
        //      helping it blend seamlessly with the soft droplets that
        //      preceded it in the animation.
        float glow = pow(max(0.0, 1.0 - distToLine * RING_GLOW_SPREAD), GLOW_FALLOFF_POWER);
        // --- The Shape Composition (The Final Merge) ---
        // Combines the solid ring "core" and the atmospheric "glow" into one mask.
        // 1. The Maximum Operator (max):
        //    - Instead of adding (+), we use `max`. This ensures that the
        //      center of the ring stays at 1.0 (fully solid) and doesn't
        //      become "over-exposed" or brighter than its intended color.
        // 2. The Intensity Weight (* GLOW_INTENSITY):
        //    - Allows for fine-tuning the brightness of the halo. If set to
        //      0.5, the glow will be subtle and ghostly; if set to 1.0,
        //      it will look like a neon light.
        // 3. Visual Depth:
        //    - This layering is what makes the ring look "premium." The `core`
        //      provides the structural clarity needed for UI, while the `glow`
        //      provides the organic "gooey" texture that matches your
        //      liquid simulation.
        alphaShape = max(core, glow * parameters.glowIntensity);
    } else {
        // --- The Droplet Core (The Liquid Body) ---
        // Defines the solid inner area of an individual falling droplet.
        // 1. Inner Radius (0.4 to 0.3):
        //    - Note that we are using much smaller values than the ring (1.0).
        //    - This keeps the "solid" part of the drop small, leaving plenty
        //      of room for the soft, gooey edges to blend with other drops.
        // 2. Soft Inversion:
        //    - By starting the smoothstep at 0.4 and ending at 0.3, we create
        //      a value that is 1.0 (opaque) at the center and fades to 0.0
        //      as it moves outward.
        // 3. The "Meta-ball" Foundation:
        //    - This core is designed to be intentionally blurry. In a liquid
        //      simulation, you don't want sharp edges here; you want a soft
        //      gradient so that when two drops get close, their alphas add
        //      together to create that "merging" or "clumping" effect.
        float core = smoothstep(0.4, 0.3, dist);
        // --- The Liquid Field (The Gooey Aura) ---
        // Creates the soft "potential field" that allows droplets to merge.
        // 1. Radial Falloff (1.0 - dist):
        //    - Since 'dist' is 0.0 at the center, this starts at 1.0 and
        //      decreases linearly toward the edge of the particle.
        // 2. The Power Curve (pow ... GLOW_FALLOFF_POWER):
        //    - This transforms the linear gradient into an exponential one.
        //    - It concentrates the "thickness" near the center while leaving
        //      a long, soft "tail" of transparency toward the edges.
        // 3. Metaball Synergy:
        //    - When two particles with this soft falloff overlap, their alpha
        //      values sum up. Once they cross a certain threshold, they
        //      appear to "snap" together, creating the illusion of
        //      surface tension and merging liquid.
        float glow = pow(max(0.0, 1.0 - dist), GLOW_FALLOFF_POWER);
        // --- The Droplet Synthesis (The Density Map) ---
        // Merges the inner body and the outer field of the falling droplet.
        // 1. The Max Operator:
        //    - Using `max` ensures that the center of the droplet remains a
        //      solid, constant density. This prevents the "nucleus" of the
        //      drop from looking like a bright, over-exposed bloom.
        // 2. The Liquid Threshold (* GLOW_INTENSITY):
        //    - This value determines how "sticky" the liquid looks. A higher
        //      intensity makes the droplets appear thicker and more likely
        //      to blob together into larger masses.
        // 3. The Result:
        //    - You now have a particle that is solid in the middle but has
        //      an "invisible" reach—a field of influence that will interact
        //      with its neighbors to create the liquid effect.
        alphaShape = max(core, glow * parameters.glowIntensity);
    }
    
    return float4(in.color.rgb, alphaShape * in.color.a);
}
