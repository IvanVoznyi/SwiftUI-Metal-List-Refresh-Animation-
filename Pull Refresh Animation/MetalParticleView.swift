import SwiftUI
import MetalKit

// --- 1. SwiftUI Wrapper ---
struct MetalParticleView: UIViewRepresentable {
    var scrollOffset: CGFloat
    var color: Color
    var triangleTop: Float
    var triangleBottom: Float
    var opacityInCircle: Float
    var glowIntensity: Float
    
    func makeUIView(context: Context) -> MTKView {
        let mtkView = MTKView()
        mtkView.device = MTLCreateSystemDefaultDevice()
        mtkView.backgroundColor = .clear
        mtkView.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        
        context.coordinator.renderer = ParticleRenderer(metalKitView: mtkView)
        return mtkView
    }
    
    func updateUIView(_ uiView: MTKView, context: Context) {
        context.coordinator.renderer?.scrollOffset = Float(scrollOffset)
        context.coordinator.renderer?.color = color
        context.coordinator.renderer?.triangleTop = triangleTop
        context.coordinator.renderer?.triangleBottom = triangleBottom
        context.coordinator.renderer?.opacityInCircle = opacityInCircle
        context.coordinator.renderer?.glowIntensity = glowIntensity
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator()
    }
    
    class Coordinator {
        var renderer: ParticleRenderer?
    }
}

// --- 2. Metal Engine ---
class ParticleRenderer: NSObject, MTKViewDelegate {
    // --- GPU Hardware Interface (The Master Key) ---
    // A direct reference to the physical Graphics Processing Unit (GPU) hardware inside the device.
    // 1. The "Factory" Pattern:
    //    - This object is the root creator for everything in Metal. You cannot create a
    //      Command Queue, Buffer, Texture, or Library without asking the 'device' to do it.
    //    - It ensures that any resource created is compatible with the specific silicon (e.g., A16 Bionic) running the app.
    // 2. The Abstraction Bridge:
    //    - Swift code runs on the CPU; graphics rendering runs on the GPU.
    //    - This variable serves as the primary communication link, allowing the CPU to allocate
    //      memory (VRAM) and issue instructions that the hardware understands.
    // 3. One-Time Initialization:
    //    - We initialize this only once (usually via `MTLCreateSystemDefaultDevice()`).
    //    - It persists for the app's lifecycle because losing the link to the GPU would
    //      destroy all graphics resources associated with it.
    var device: MTLDevice!
    // --- GPU Work Submission Pipeline (The Order Wheel) ---
    // A serialized queue that organizes and submits work to the GPU for execution.
    // 1. The "Traffic Controller":
    //    - The GPU is a separate processor that runs asynchronously (independently) from the CPU.
    //    - You cannot just tell it "Draw now!" because it might be busy rendering the previous frame.
    //    - This queue holds the list of tasks (Command Buffers) in a strict First-In-First-Out (FIFO) order.
    // 2. The Workflow Relationship:
    //    - The `device` builds the factory.
    //    - The `commandQueue` manages the assembly line.
    //    - Every frame, we ask this queue to create a new `MTLCommandBuffer` (a blank work order),
    //      fill it with drawing instructions, and then `commit` it back to the queue.
    // 3. Efficiency (No Stalling):
    //    - By using a queue, the CPU can prepare the instructions for the *next* frame while the
    //      GPU is still busy drawing the *current* frame.
    //    - This parallelism is what allows games and animations to run at a smooth 60 or 120 FPS.
    var commandQueue: MTLCommandQueue!
    // --- Compiled Compute Kernel State (The Physics Engine) ---
    // A highly optimized, executable object containing the machine code for a specific compute function.
    // 1. The "Brain" of the Simulation:
    //    - This object represents the compiled version of your "updateParticles" kernel function.
    //    - It contains the actual instructions (add, multiply, sin, cos) that thousands of GPU threads
    //      will execute simultaneously to calculate particle movement.
    // 2. The "Expensive" Creation:
    //    - Creating this state involves compiling the human-readable Metal code into raw GPU machine code.
    //    - This process is slow (taking milliseconds), so we do it strictly once during setup (`init`).
    //    - During the render loop, we simply "bind" this pre-cooked state, which is instant.
    // 3. The Immutable Snapshot:
    //    - Once created, the pipeline state cannot be changed.
    //    - It locks in critical optimization decisions made by the driver (like register usage and thread grouping)
    //      specific to the hardware (e.g., A-series vs. M-series chips).
    var computeState: MTLComputePipelineState!
    // --- Compiled Graphics Pipeline State (The Artist) ---
    // A highly optimized, immutable object that defines the complete configuration for a specific rendering pass.
    // 1. The "Fusion" of Shaders:
    //    - While the Compute State runs a single function, the Render State acts as the "glue"
    //      that binds two distinct stages together:
    //      - The Vertex Function (Positioning geometry).
    //      - The Fragment Function (Coloring pixels).
    // 2. The Fixed-Function Settings:
    //    - It locks in critical hardware settings that shaders cannot change on the fly, such as:
    //      - Blending Mode (e.g., Additive vs. Alpha Blending).
    //      - Pixel Format (e.g., .bgra8Unorm) to ensure color compatibility with the screen.
    // 3. Performance Strategy:
    //    - Like the Compute State, this object is computationally expensive to create (involving
    //      driver compilation and validation).
    //    - We build it once during `init` and simply reuse it every frame. This allows the GPU to
    //      instantly switch "drawing modes" without stalling.
    var renderState: MTLRenderPipelineState!
    // --- Primary Particle Storage (The VRAM Array) ---
    // A reference to a contiguous block of GPU memory that holds the state of every particle.
    // 1. The "Source of Truth":
    //    - Swift arrays exist in CPU RAM. The GPU cannot read them directly or fast enough.
    //    - This buffer lives in VRAM (Video RAM), optimized for massive parallel read/write access.
    //    - It stores an array of 'Particle' structs (Position + Velocity + Color + Size).
    // 2. The Shared Resource:
    //    - The Compute Shader reads the current positions, calculates physics, and writes NEW values back here.
    //    - The Render Shader immediately reads those updated positions to draw the dots on screen.
    // 3. Persistence:
    //    - Without this buffer, the GPU would forget where particles are the moment a frame finishes rendering.
    //    - This memory block persists across frames, allowing the simulation to evolve continuously over time.
    var particleBuffer: MTLBuffer!
    
    // --- Physics Shape Parameter (Funnel Geometry) ---
    // Controls the horizontal spread of particles at the very top of the screen (the "Wide Mouth").
    // 1. The Funnel Concept:
    //    - The particles are not just falling in a straight box; they are confined to a triangular shape.
    //    - This value determines how wide the triangle is at the spawn point (Y = Top).
    // 2. The Visual Effect:
    //    - 1.0 = The particles spread exactly across the screen width (edges touch).
    //    - 1.2 = The particles spread slightly wider than the screen (off-screen spawning).
    //      This creates a seamless "infinite rain" effect where you don't see the hard edges.
    //    - 0.5 = The particles spawn in a tight column in the center.
    // 3. Dynamic Usage:
    //    - This value is passed to the GPU every frame via the `Uniforms` struct.
    //    - The Compute Shader uses it to calculate the `mix()` factor for the horizontal position (`hX`).
    var triangleTop: Float = 1.2
    // --- Physics Shape Parameter (Funnel Tip) ---
    // Controls the horizontal spread of particles at the bottom of the screen (near the circle).
    // 1. The Convergence Point:
    //    - As particles fall, they don't move in straight vertical lines.
    //    - They interpolate (mix) from the wide `triangleTop` width down to this narrower width.
    // 2. The Visual Effect (Focusing):
    //    - 0.25 (25% Width): This forces the particles to "aim" for the center, creating a funnel or V-shape.
    //    - If this were 1.0 (matching the top), it would look like standard rain (a rectangle).
    //    - If this were 0.0, all particles would converge to a single pixel point at the bottom.
    // 3. GPU Calculation:
    //    - The Compute Shader uses this value in the `mix()` function to determine the exact
    //      X-position based on the particle's current Y-height (progress).
    var triangleBottom: Float = 0.25

    // --- Target Physics State (The Input) ---
    // The raw destination value received from the SwiftUI view (user gesture).
    // 1. The "Magnet":
    //    - This acts as the target point for the physics simulation.
    //    - The `internalOffset` (the actual position of the animation) constantly tries to
    //      "chase" or catch up to this value using the friction logic we wrote.
    // 2. The Interaction:
    //    - 0.0: The user is at the top of the list (rest state).
    //    - > 0.0: The user is pulling down. The larger the number, the stronger the "Pull" effect.
    // 3. The Signal:
    //    - This value is eventually normalized (divided by 75.0 or 100.0) in the shader to drive
    //      the 0.0 to 1.0 progress of the Morph Phase (Dot -> Ring).
    var scrollOffset: Float = 0
    
    // --- Global Animation Clock (The Heartbeat) ---
    // Tracks the total elapsed seconds since the animation began.
    // 1. The "Pulse":
    //    - While 'scrollOffset' drives the interactive parts (Morphing), 'time' drives the
    //      continuous, automatic parts (Wiggling, Pulsing, glowing).
    //    - Even if the user stops scrolling, the particles keep moving because this clock keeps ticking.
    // 2. The Shader Driver:
    //    - This value is passed to the GPU every frame via the `Uniforms` struct.
    //    - Shaders use it inside trigonometric functions like `sin(time * speed)` to create
    //      smooth, repeating loops (oscillations) for the "breathing" ring effect.
    // 3. Precision Note:
    //    - We use `Float` (32-bit) instead of `Double` (64-bit) because GPUs are optimized for
    //      32-bit math. Sending a Double would require conversion or waste bandwidth.
    var time: Float = 0
    // --- The Reference Epoch (Time Anchor) ---
    // Stores the system timestamp exactly when the renderer initialized.
    // 1. The Problem with System Time:
    //    - `CACurrentMediaTime()` returns the number of seconds since the device last rebooted
    //      (e.g., 45,000.0 seconds).
    //    - If we passed this huge number directly to the GPU, `sin(45000.0)` would lose precision
    //      due to floating-point rounding errors, causing the animation to jitter or "shake."
    // 2. The Solution (Relative Time):
    //    - By subtracting `startTime` from the current time every frame (`current - start`),
    //      we ensure our animation clock starts at exactly `0.0`.
    //    - This keeps the numbers small and manageable for the 32-bit floats used in shaders.
    // 3. High Precision:
    //    - We use `CFTimeInterval` (Double) here on the CPU to maintain maximum accuracy
    //      before casting down to `Float` for the GPU.
    var startTime: CFTimeInterval = 0
    // --- Visual Theme Parameter (The Paint) ---
    // Stores the high-level SwiftUI color that will tint the entire particle system.
    // 1. The Bridge (SwiftUI -> Metal):
    //    - Metal does not understand what a "Color" or "UIColor" object is.
    //    - This variable captures the user's intent (e.g., .blue) so we can later convert it
    //      into raw numbers (Red, Green, Blue, Alpha) that the GPU math units can process.
    // 2. The Uniform:
    //    - This color is passed to the shader as a "Uniform," meaning every single particle
    //      uses this same base color.
    //    - The shader then modifies its transparency (Alpha) individually based on physics.
    // 3. Dynamic Updates:
    //    - Because this is a `var`, the color can respond to system changes (like Dark Mode)
    //      or user themes instantly without needing to restart the renderer.
    var color: Color = .blue
    var opacityInCircle: Float = 0.3
    var glowIntensity: Float = 0.6

    // --- System Density Configuration (The Population) ---
    // Defines the strict, immutable number of dots simulated in the scene.
    // 1. The "Goldilocks" Number:
    //    - 250 is chosen specifically for visual balance.
    //    - Too few (e.g., 50): The "Ring" stage looks like a dotted line with gaps.
    //    - Too many (e.g., 1000): The "Falling" stage looks like a messy blizzard, losing the elegance.
    // 2. The Dual Role (The Trick):
    //    - In our shader logic, this isn't just a count.
    //    - Indices 0 to 248 are treated as "Falling Dots."
    //    - Index 249 (The last one) is hijacked by the shader to become the "Solid Ring."
    //    - This allows us to draw two completely different objects (dots and a ring) in a single draw call.
    // 3. Buffer Allocation:
    //    - This number dictates exactly how much memory we reserve in VRAM during `init`.
    //    - It must match the logic in the shader (e.g., if the shader expects a specific ring index).
    let particleCount = 251
    
    // --- Physics Simulation State (The Follower) ---
    // The actual, smoothed position of the animation used for rendering.
    // 1. The Decoupling Strategy:
    //    - We separate the "Input" (`scrollOffset`) from the "Output" (`internalOffset`).
    //    - `scrollOffset` is the raw, jerky target where the user's finger is.
    //    - `internalOffset` is the "Ghost" that chases the target using physics.
    // 2. The "Rubber Band" Effect:
    //    - By updating this value gradually every frame (instead of snapping to the finger),
    //      we create the feeling of weight and friction.
    //    - This is what makes the pull feel "organic" rather than robotic.
    // 3. The Driver:
    //    - This is the final value sent to the GPU.
    //    - If the user stops scrolling abruptly, this variable continues to move until it
    //      settles, creating that nice "drift" effect.
    var internalOffset: Float = 0
    // --- Physics Timing State (The Anchor) ---
    // Stores the exact timestamp of the previous frame to calculate motion delta.
    // 1. The "Delta Time" Necessity:
    //    - To move objects smoothly at different frame rates (60Hz vs 120Hz), we need to know
    //      exactly how much time passed since the last draw call.
    //    - `lastTime` holds the "Start" of that interval. `currentTime` holds the "End".
    // 2. The Logic:
    //    - `dt = currentTime - lastTime`.
    //    - If `dt` is 0.016 (16ms), we move the object 16% of its speed.
    //    - If `dt` is 0.033 (33ms, a lag spike), we move it 33% to catch up.
    // 3. The Reset:
    //    - Crucially, we update this variable at the very end of the draw loop (`lastTime = currentTime`).
    //    - This ensures the next frame calculates the difference relative to *now*.
    var lastTime: CFTimeInterval = 0
    
    // --- Physics Constraint (The Speed Limit) ---
    // Defines the maximum velocity at which the internal animation is allowed to chase the user's finger.
    // 1. The "Lag" Creator:
    //    - By capping the speed, we prevent the particles from snapping instantly to the
    //      scroll position. This forces them to "drag" behind, creating a feeling of weight.
    // 2. The Formula (Distance / Time):
    //    - 150.0: The distance in points (pixels).
    //    - 1.5: The duration in seconds.
    //    - Result: "Travel 100 points per second."
    // 3. The Usage:
    //    - Inside the update loop, we multiply this by `dt` (time elapsed) to calculate the
    //      maximum step size for that specific frame.
    //    - This ensures consistent speed regardless of frame rate (60fps vs 120fps).
    let maxSpeedPerSecond: Float = 150.0 / 1.5
    
    // --- Data Structure Contract (The Blueprint) ---
    // A Swift representation of the exact memory layout expected by the GPU shader.
    // 1. The Mirror Image:
    //    - This struct must match the C++ `struct Particle` in your Metal file byte-for-byte.
    //    - If you add a float here but forget to add it in the shader (or vice versa),
    //      the GPU will read the wrong bytes, causing glitches or crashes.
    // 2. SIMD Types (Single Instruction, Multiple Data):
    //    - We use `SIMD2<Float>` instead of `CGPoint` or `CGSize`.
    //    - SIMD types are optimized for vector math on modern CPUs and map directly to
    //      Metal's vector types (`float2`, `float3`, `float4`).
    // 3. Memory Alignment (The Padding):
    //    - GPUs process data in chunks (usually 16 bytes).
    //    - If our struct size isn't a "nice" number (like 32 or 48 bytes), the GPU memory controller
    //      might get confused about where the next particle starts.
    //    - `pad` is a "dummy" variable used to force the total size of the struct to align
    //      perfectly with these hardware requirements.
    struct Particle {
        var pos: SIMD2<Float>
        var col: SIMD4<Float>
        var size: Float
        var pad: Float = 0
    }
    
    // 1. System Data (Matches Metal Uniforms)
    struct Uniforms {
        var screenSize: SIMD2<Float>
        var scrollOffset: Float
        var time: Float
    }

    // 2. User Data (Matches Metal Parameters)
    struct Parameters {
        var color: SIMD4<Float>
        var particleCount: Int32
        var triangleTop: Float
        var triangleBottom: Float
        var opacityInCircle: Float
        var glowIntensity: Float
    }

    init?(metalKitView: MTKView) {
        // --- Initialize Parent Foundation (The Base) ---
        // Calls the designated initializer of the superclass (NSObject) to prepare the object instance.
        // 1. The Hierarchy:
        //    - Our class `ParticleRenderer` inherits from `NSObject`.
        //    - Before we can customize it with Metal-specific tools (Device, Queue, Pipelines),
        //      we must first ensure the basic object machinery (memory allocation, identity) is set up.
        // 2. The Chain of Command:
        //    - Swift enforces strict initialization rules.
        //    - Calling `super.init()` guarantees that all properties inherited from the parent
        //      are in a valid state before we start using `self`.
        // 3. Runtime Registration:
        //    - This step registers our new instance with the Objective-C runtime, allowing it to
        //      act as a delegate for `MTKView` later on.
        super.init()

        // --- Connect to GPU Hardware (The Handshake) ---
        // Attempts to acquire a reference to the default graphics processing unit.
        // 1. The Entry Point:
        //    - This function is the "Big Bang" of any Metal app.
        //    - It asks the operating system: "Give me the primary GPU so I can talk to it."
        // 2. Hardware Abstraction:
        //    - Whether running on an iPhone (A-series) or a Mac (M-series), this function
        //      returns a unified `MTLDevice` interface, hiding the complex hardware differences.
        // 3. The Safety Check (Guard):
        //    - In rare cases (e.g., extremely old simulators or corrupted states), this might fail.
        //    - If we can't get a GPU, the renderer is useless. We return `nil` immediately
        //      to prevent the app from crashing later when we try to use `device`.
        guard let device = MTLCreateSystemDefaultDevice() else { return nil }
        // --- Persist the GPU Connection (The Anchor) ---
        // Stores the local device reference into our class property for long-term use.
        // 1. Scope Extension:
        //    - The `device` variable created above is local to this `init` function. It dies when the function ends.
        //    - By assigning it to `self.device`, we keep the connection alive for the entire lifespan of the renderer.
        // 2. The Factory Role:
        //    - We will need this object later in `draw()` to create buffers and textures.
        //    - It acts as the "Mother" object that spawns all other GPU resources.
        // 3. Ownership:
        //    - This assignment increments the reference count, ensuring the system doesn't accidentally
        //      destroy the GPU interface while we are still using it.
        self.device = device
        // --- Open the GPU Submission Channel (The Assembly Line) ---
        // Asks the GPU device to create a new queue for organizing and executing commands.
        // 1. The Serialization Hub:
        //    - The GPU can do many things at once, but it needs an ordered list of tasks to stay efficient.
        //    - This queue acts as the "Traffic Controller," ensuring that our physics calculations
        //      (Compute) and our drawing instructions (Render) happen in the correct sequence.
        // 2. Resource Management:
        //    - Creating a Command Queue is an expensive setup step. We do it once here and reuse it
        //      every single frame to avoid the overhead of opening a new channel 60 or 120 times per second.
        // 3. Asynchronous Flow:
        //    - This queue allows the CPU to quickly "fire and forget" commands into the GPU's
        //      on-board memory. This prevents the CPU from having to wait for the GPU to finish
        //      painting pixels before it can start calculating the next frame.
        self.commandQueue = device.makeCommandQueue()

        // --- Linking the View to the Hardware (The Display Bridge) ---
        // Informs the MTKView which physical GPU it should use to manage its internal textures.
        // 1. The Configuration Requirement:
        //    - An MTKView is a high-level SwiftUI/UIKit wrapper, but it cannot display anything
        //      until it knows which hardware chip is responsible for "painting" its pixels.
        //    - By setting this property, you authorize the view to allocate the "Back Buffer"
        //      (the hidden canvas) on this specific GPU.
        // 2. Resource Synchronization:
        //    - This ensures that the `device` used by our `ParticleRenderer` is the same `device`
        //      powering the view.
        //    - If they didn't match (e.g., using a discrete GPU for math but an integrated GPU for display),
        //      the app would crash when trying to share memory between them.
        // 3. Automation:
        //    - Once this is set, the MTKView can automatically manage its `currentDrawable` and
        //      `currentRenderPassDescriptor`, which we use every frame to draw the particles.
        metalKitView.device = device
        // --- Establishing the Frame Loop (The "Action" Contract) ---
        // Designates our ParticleRenderer class as the authoritative controller for the view's update cycle.
        // 1. The Delegation Pattern:
        //    - MTKView is "smart" enough to know when the screen needs to refresh, but it is
        //      "blind" to what needs to be drawn.
        //    - By setting `delegate = self`, we are signing a contract saying: "Whenever the screen
        //      is ready for a new frame, call the functions inside THIS class to do the work."
        // 2. Triggering the Draw Loop:
        //    - This is the "On" switch for the animation.
        //    - Once this line executes, the MTKView will start calling our `draw(in:)` function
        //      automatically at the device's native refresh rate (60Hz or 120Hz).
        // 3. Separation of Concerns:
        //    - This keeps the SwiftUI view wrapper (MetalParticleView) clean.
        //    - The view handles the layout, while this delegate (self) handles the heavy
        //      GPU lifting and frame-by-frame physics logic.
        metalKitView.delegate = self
        // --- Setting the Absolute Time Anchor (The Birth of the Scene) ---
        // Records the exact system timestamp at the moment the simulation is born.
        // 1. Monotonic Time Security:
        //    - `CACurrentMediaTime()` is based on the 'mach_absolute_time' of the CPU.
        //    - Unlike `Date()`, which can jump backward if the user changes the system clock,
        //      this timer is "monotonic"—it only ever goes forward. This is critical for
        //      preventing glitches in your physics calculations.
        // 2. Solving the Precision Problem:
        //    - The system clock might return a massive number (e.g., 54,230.5 seconds since boot).
        //    - GPUs use 32-bit floats, which lose accuracy when numbers get too high.
        //    - By saving this `startTime`, we can calculate a "Relative Time" (Current - Start).
        //      This keeps the `time` variable starting at 0.0, ensuring perfectly smooth math
        //      for wiggles and pulses.
        // 3. The One-Time Snapshot:
        //    - We capture this once during initialization so we always have a fixed
        //      "Zero Point" to look back at, regardless of how long the app stays open.
        self.startTime = CACurrentMediaTime()
        
        // --- Accessing the Shader Repository (The Blueprint Library) ---
        // Retrieves the collection of pre-compiled GPU functions bundled within the app.
        // 1. The "The Catalog":
        //    - When you build your project in Xcode, all your '.metal' files are compiled
        //      into a single file called 'default.metallib'.
        //    - This line opens that file and creates a searchable index of all your
        //      Compute, Vertex, and Fragment shaders.
        // 2. Resource Discovery:
        //    - Without this 'lib' object, the CPU has no way to find the GPU code you wrote.
        //    - It acts as the "Phonebook" that we will use in the next steps to look up
        //      specific functions like "updateParticles" or "particleVertex" by their string names.
        // 3. Safety Check:
        //    - If this returns 'nil', it usually means there are no Metal files in your
        //      target or the compilation failed.
        //    - We store it in a local constant because we only need it briefly to
        //      extract the functions we need to "bake" our Pipeline States.
        let lib = device.makeDefaultLibrary()
        // --- Extracting the Compute Logic (The Mathematician) ---
        // Searches the library for the specific entry point used for physics calculations.
        // 1. Function Mapping:
        //    - This looks for a kernel marked `kernel void updateParticles(...)` in your .metal file.
        //    - This function is responsible for the "invisible" work: calculating where
        //      dots should fall, how they wiggle, and how they morph into the ring.
        // 2. Parallel Blueprint:
        //    - Unlike standard Swift functions, this represents a "Massively Parallel" instruction
        //      set designed to be executed by hundreds of GPU cores simultaneously.
        let compFunc = lib?.makeFunction(name: "updateParticles")
        // --- Extracting the Vertex Logic (The Architect) ---
        // Retrieves the function responsible for geometric positioning on the screen.
        // 1. Coordinate Transformation:
        //    - This maps to `vertex VertexOut particleVertex(...)`.
        //    - Its primary job is to take the "World" positions (calculated by the compute shader)
        //      and project them into "Clip Space" (the 2D coordinate system of your phone screen).
        // 2. Sizing and Metadata:
        //    - It also passes along critical data like the `[[point_size]]`, telling the GPU
        //      how physically large each particle should be rendered.
        let vertFunc = lib?.makeFunction(name: "particleVertex")
        // --- Extracting the Fragment Logic (The Painter) ---
        // Retrieves the function that determines the final color and shape of every pixel.
        // 1. The Rasterization Gate:
        //    - This maps to `fragment float4 particleFragment(...)`.
        //    - While the Vertex shader handles "Where," this handles "What." It paints the
        //      soft glows, the sharp dots, and the transparency of the ring.
        // 2. Procedural Drawing:
        //    - We use this function to calculate "Distance from Center" to draw perfectly
        //      round circles and glowing halos mathematically, rather than using images.
        let fragFunc = lib?.makeFunction(name: "particleFragment")
        
        do {
            // --- Compiling the Compute Pipeline (The Engine Build) ---
            // Transforms the high-level 'updateParticles' function into an executable GPU "State."
            // 1. The "Baking" Process:
            //    - Simply having the function code isn't enough. The GPU needs to "compile" that
            //      code into a hardware-specific binary optimized for your device's specific chip.
            //    - This is an expensive operation, which is why we do it once during 'init' and
            //      store it in 'computeState' for reuse.
            // 2. Hardware Resource Planning:
            //    - During this call, the system calculates how many GPU registers and how much
            //      thread-group memory the shader will need.
            //    - This allows the GPU to manage its massive parallel workload efficiently when
            //      we later tell it to update 250 particles at once.
            // 3. The Error Threshold:
            //    - We use 'try' because this is a point of failure. If the shader code has a
            //      logic error or hardware incompatibility, the "State" creation will fail.
            //    - Once successful, this 'computeState' acts as the "Command Center" for all
            //      physics calculations in our simulation.
            self.computeState = try device.makeComputePipelineState(function: compFunc!)
            
            // --- Designing the Drawing Blueprint (The Spec Sheet) ---
            // Creates a configuration object used to define how the GPU should render pixels.
            // 1. The "Pre-Flight" Checklist:
            //    - A Render Pipeline is complex; it needs to know which Vertex function to use,
            //      which Fragment function to use, and how the colors should blend together.
            //    - Think of this 'Descriptor' as a form you fill out to tell the GPU exactly
            //      how you want your "Drawing Machine" to be built.
            // 2. Mutable Configuration:
            //    - This object is temporary. We use it to set our preferences (like vertex/fragment
            //      links and pixel formats) before we "freeze" it into a permanent,
            //      high-performance Pipeline State.
            // 3. Centralizing the Pipeline:
            //    - By using a descriptor, we ensure that all the different stages of the
            //      graphics pipeline (Positioning -> Rasterization -> Coloring) are
            //      perfectly synchronized and compatible with each other.
            let rDesc = MTLRenderPipelineDescriptor()
            
            // --- Connecting the Geometry Stage (The Landmark) ---
            // Attaches the compiled vertex function to the render pipeline blueprint.
            // 1. Point of Entry:
            //    - This tells the GPU that for every particle we draw, it must first run
            //      the 'particleVertex' code to determine the dot's size and screen position.
            // 2. Data Flow:
            //    - The vertex function acts as a bridge; it takes the raw particle data and
            //      prepares it for the "Rasterizer," which turns mathematical points into
            //      actual clusters of pixels.
            rDesc.vertexFunction = vertFunc
            // --- Connecting the Color Stage (The Aesthetic) ---
            // Attaches the compiled fragment function to the render pipeline blueprint.
            // 1. Pixel-Level Control:
            //    - This tells the GPU to run the 'particleFragment' code for every single
            //      pixel covered by a particle.
            // 2. Final Output:
            //    - While the vertex function handles "Where," this handles "What color."
            //      It is here that the glow, the transparency, and the specific tint of
            //      the particles are calculated before they hit the screen.
            rDesc.fragmentFunction = fragFunc
            
            // --- Synchronizing the Canvas Format (The Color Agreement) ---
            // Tells the pipeline exactly how to "write" color bits to the screen's memory.
            // 1. The Data Signature:
            //    - Pixel formats (like BGRA8Unorm) define the order and size of the Red, Green,
            //      Blue, and Alpha channels in memory.
            //    - If the Pipeline and the View don't agree on this format, the GPU will
            //      essentially be "speaking a different language," resulting in a crash
            //      or a corrupted, garbled display.
            // 2. Automated Compatibility:
            //    - By fetching the format directly from the 'metalKitView' rather than hard-coding
            //      it, we ensure the renderer works perfectly regardless of the device's
            //      hardware (e.g., standard displays vs. Wide Color P3 displays).
            // 3. The Final Handshake:
            //    - This is the final step in ensuring the "Render Pipeline" we are building is
            //      ready to pour its finished pixels into the view's "Drawable" texture.
            rDesc.colorAttachments[0].pixelFormat = metalKitView.colorPixelFormat
            // --- Activating the Transparency Engine (The Compositor) ---
            // Enables the mathematical mixing of new pixels with existing background pixels.
            // 1. Beyond "Opaque" Rendering:
            //    - By default, the GPU simply overwrites whatever was previously on the screen
            //      with the new color (Opaque mode).
            //    - Enabling blending allows the GPU to perform "Alpha Compositing," which is
            //      essential for rendering things that are see-through or glowing.
            // 2. The Visual Necessity:
            //    - In this particle system, our dots have soft, anti-aliased edges and
            //      the ring has a glowing halo.
            //    - Without this line, the "clear" parts of your particle squares would appear
            //      as solid black or white boxes instead of being transparent.
            // 3. Performance Note:
            //    - While blending requires the GPU to do more work (it has to "Read" the
            //      background before "Writing" the new color), it is necessary for the
            //      organic, high-end feel of this SwiftUI pull-to-refresh effect.
            rDesc.colorAttachments[0].isBlendingEnabled = true
            // --- Defining the Input Influence (The Source Factor) ---
            // Determines how much of the "New" particle's color contributes to the final pixel.
            // 1. The Blending Equation:
            //    - Graphics math follows the formula: (Source * Factor) + (Destination * Factor).
            //    - By setting this to `.sourceAlpha`, we tell the GPU: "Multiply the color of the
            //      incoming particle by its own transparency value (Alpha)."
            // 2. Variable Opacity:
            //    - If a particle's Alpha is 1.0 (Opaque), it contributes 100% of its color.
            //    - If the Alpha is 0.2 (Faded), it only contributes 20% of its color.
            // 3. Visual Softness:
            //    - This is the secret to the "Anti-Aliased" look of your dots.
            //    - The pixels at the very edge of a circle have a lower Alpha, so they contribute
            //      less color, creating a smooth transition rather than a jagged edge.
            rDesc.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
            // --- Defining the Background Retention (The Destination Factor) ---
            // Determines how much of the "Existing" screen color is kept when a particle is drawn.
            // 1. The Balancing Act:
            //    - This works in tandem with the 'sourceAlpha' factor to ensure the total
            //      brightness of the pixel stays consistent.
            //    - `.oneMinusSourceAlpha` tells the GPU: "Take whatever space is NOT occupied
            //      by the particle and show the background through it."
            // 2. The Transparency Math:
            //    - If your particle is 30% opaque (0.3), this factor becomes 70% (1.0 - 0.3).
            //    - The GPU calculates: (30% Particle Color) + (70% Background Color).
            // 3. Achieving the "Glass" Effect:
            //    - This is what allows your particles to look like they are floating *over* //      your SwiftUI list content rather than just cutting a hole in it.
            //    - It creates the perfect "standard" transparency used in almost all
            //      modern UI design.
            rDesc.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
            
            //--- Blending Configuration (The Mixer) ---
            // This line configures the math for how the transparency (Alpha) of the NEW pixel
            // mixes with what is already on the screen.
            // 1. colorAttachments[0]:
            //    - We are configuring the first (and usually only) render target—the screen.
            // 2. sourceAlphaBlendFactor:
            //    - This targets the "Source" (the particle we are currently drawing).
            //    - We are defining what we multiply the incoming Alpha by before adding it.
            // 3. .sourceAlpha:
            //    - The multiplier is the alpha value itself.
            //    - The Math: FinalAlpha contribution = (SourceAlpha * SourceAlpha).
            //    - The Visual Result: This "squares" the transparency.
            //      - A value of 1.0 (opaque) stays 1.0.
            //      - A value of 0.5 (semi-transparent) becomes 0.25 (mostly transparent).
            //      - This makes the soft edges of your particles fade away much faster,
            //        helping to sharpen the definition of the "liquid."
            rDesc.colorAttachments[0].sourceAlphaBlendFactor = .sourceAlpha
            //--- Blending Configuration (The Backdrop) ---
            // This line configures the "Destination"—the pixel already sitting on the canvas.
            // It decides how much of the old alpha value should remain after drawing the new particle.
            // 1. destinationAlphaBlendFactor:
            //    - This targets the "Destination" (what was already on screen before this draw).
            // 2. .oneMinusSourceAlpha:
            //    - The multiplier is (1.0 - Incoming Particle Alpha).
            //    - The Math: Background Contribution = ExistingAlpha * (1.0 - SourceAlpha).
            // 3. The Visual Logic:
            //    - This is the standard "Over" blending logic.
            //    - If the new particle is fully opaque (1.0), the background is wiped out (multiplied by 0).
            //    - If the new particle is transparent (0.0), the background is untouched (multiplied by 1).
            //    - This ensures that as your liquid accumulates, it properly covers up whatever is behind it.
            rDesc.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
            
            // --- Finalizing the Graphics Engine (The State Bake) ---
            // Freezes the configuration from the descriptor into a high-performance,
            // immutable Pipeline State.
            // 1. The Compilation Bridge:
            //    - Up until now, 'rDesc' was just a wish list of settings. This line
            //      triggers the actual "compilation" where the GPU drivers translate
            //      your Vertex and Fragment shaders into hardware-specific machine code.
            // 2. Pre-Optimized Execution:
            //    - Because this compilation happens once during 'init', the GPU doesn't
            //      have to figure out how to draw your particles every frame.
            //    - It already has the "machine" built and ready to go, which is why
            //      Metal is so much faster than older graphics APIs.
            // 3. Error Handling:
            //    - We use 'try' because this is the most common place for "Silent Failures"
            //      to become "Loud Crashes."
            //    - If the shaders are incompatible with the blending factors or the
            //      pixel format, the GPU will refuse to build the pipeline here.
            self.renderState = try device.makeRenderPipelineState(descriptor: rDesc)
            
            // --- Calculating the Memory Footprint (The Reservation) ---
            // Determines the exact number of bytes required to store the entire particle array.
            // 1. Stride vs. Size:
            //    - We use `stride` instead of `size` because `stride` includes the "padding"
            //      necessary to align the next element in an array.
            //    - If we only used `size`, the particles would be "crammed" together, and the
            //      GPU would misread the memory addresses of every particle after the first one.
            // 2. Linear Memory Allocation:
            //    - The GPU doesn't understand Swift "Arrays" or "Objects"; it only understands
            //      a continuous "Blob" of bytes.
            //    - By multiplying the stride of a single `Particle` by the total `particleCount`,
            //      we find the exact length of the "Blob" we need to request from the system RAM.
            // 3. Preparation for the Buffer:
            //    - This calculation is a prerequisite for `device.makeBuffer`. We are essentially
            //      measuring the "furniture" so we know exactly how big of a "moving truck"
            //      to rent from the GPU.
            let size = MemoryLayout<Particle>.stride * particleCount
            // --- Allocating GPU-Accessible Memory (The Shared Reservoir) ---
            // Creates a dedicated block of memory to store all particle data.
            // 1. The "VRAM" bridge:
            //    - A Buffer is essentially an array that lives on the GPU. Unlike a standard
            //      Swift array, this memory is "wired," meaning the GPU can read from and
            //      write to it at lightning speeds without constant CPU intervention.
            // 2. Storage Mode (.storageModeShared):
            //    - This is the "Shared" memory model. It allows both the CPU (Swift code)
            //      and the GPU (Metal shaders) to look at the same physical memory at the same time.
            //    - This is highly efficient for mobile devices (Apple Silicon), as it avoids
            //      copying data back and forth between different memory chips.
            // 3. The Lifecycle:
            //    - We allocate this buffer once during setup. During the simulation, the
            //      Compute shader will update the particle positions inside this buffer,
            //      and the Vertex shader will immediately read those new positions to
            //      draw them—all within the same block of memory.
            self.particleBuffer = device.makeBuffer(length: size, options: .storageModeShared)
        } catch {
            print("Metal Setup Error: \(error)")
        }
    }
    
    func getSIMDColor(from color: Color) -> SIMD4<Float> {
        let uiColor = UIColor(color)
        var r: CGFloat = 0, g: CGFloat = 0, b: CGFloat = 0, a: CGFloat = 0
        uiColor.getRed(&r, green: &g, blue: &b, alpha: &a)
        return SIMD4<Float>(Float(r), Float(g), Float(b), Float(a))
    }
    
    func draw(in view: MTKView) {
        // --- The Frame Readiness Check (The Green Light) ---
        // Ensures all hardware resources and pipeline states are perfectly aligned before
        // starting the heavy work of the frame.
        // 1. The Drawable & Descriptor (The Canvas):
        //    - 'currentDrawable' is the actual texture waiting to be shown on screen.
        //    - 'currentRenderPassDescriptor' contains the "Load Actions" (like clearing the
        //      screen to transparent).
        //    - If the screen is resizing or the app is in the background, these might be
        //      missing; we exit early to avoid "drawing into a void."
        // 2. State Validation (The Logic):
        //    - We verify that our 'computeState' (physics) and 'renderState' (visuals)
        //      finished compiling correctly during initialization.
        // 3. Command Buffer (The Shipping Container):
        //    - We create a fresh 'commandBuffer' for this specific frame.
        //    - This is the "envelope" where we will pack all our Compute and Render
        //      instructions before sending them to the GPU in one single batch.
        // 4. Thread Safety:
        //    - By using 'guard', we guarantee that the rest of the function can safely
        //      force-unwrap these variables, keeping the core rendering logic clean
        //      and free of messy "if-let" nesting.
        guard let drawable = view.currentDrawable,
              let renderPassDescriptor = view.currentRenderPassDescriptor,
              let computeState = computeState,
              let renderState = renderState,
              let particleBuffer = particleBuffer,
              let commandBuffer = commandQueue.makeCommandBuffer() else { return }
        
        // --- Capturing the Present Moment (The Frame Timestamp) ---
        // Retrieves the precise system time at the start of the current frame's execution.
        // 1. Consistency Across the Frame:
        //    - We capture the time once at the top of the 'draw' function and store it.
        //    - This ensures that every calculation performed during this frame—whether
        //      it's the physics in the Compute shader or the glow in the Fragment
        //      shader—uses the exact same "Now" value, preventing jitter.
        // 2. High-Precision Animation:
        //    - This value is measured in seconds (including fractions) since the device
        //      booted up.
        //    - Because it is highly precise, it allows us to create silky-smooth
        //      interpolations for the particle movements, even if the frame rate
        //      fluctuates slightly between 60Hz and 120Hz (ProMotion).
        // 3. Time Delta Preparation:
        //    - We will use this 'currentTime' relative to our 'startTime' to tell
        //      the GPU how many seconds have passed since the animation began,
        //      driving the evolution of the "liquid" effect.
        let currentTime = CACurrentMediaTime()
        // --- Establishing the First Delta (The Initial Synchronizer) ---
        // Ensures the physics engine doesn't experience a "Time Jump" on the very first frame.
        // 1. Preventing the Physics Explosion:
        //    - Physics engines calculate movement based on "Delta Time" (the time elapsed
        //      since the last frame).
        //    - If we didn't set 'lastTime' on frame #1, the first calculation would subtract
        //      'currentTime' (e.g., 50,000s) from 0, resulting in a massive leap that
        //      would teleport your particles off-screen instantly.
        // 2. The One-Time Bootstrap:
        //    - This 'if' check only passes once in the entire lifecycle of the renderer.
        //    - It essentially says: "On the first frame, the 'previous' time is just 'now,'
        //      so the time elapsed is effectively zero."
        // 3. Ensuring Smooth Continuity:
        //    - By anchoring the clock here, we guarantee that the simulation begins
        //      with a tiny, manageable step, allowing the liquid animation to
        //      start from a resting state rather than a chaotic burst.
        if lastTime == 0 { lastTime = currentTime }
        // --- Measuring the Frame Interval (The Heartbeat) ---
        // Calculates the exact duration of the gap between the previous frame and this one.
        // 1. Frame-Rate Independence:
        //    - Not every frame takes exactly 1/60th of a second. If the CPU gets busy, a
        //      frame might take longer.
        //    - By calculating this `rawDelta`, we can scale our physics movement. If a
        //      frame is twice as long, we move the particles twice as far, ensuring the
        //      animation speed looks identical to the human eye regardless of lag.
        // 2. The Bridge to the GPU:
        //    - `CACurrentMediaTime` returns a `Double` for extreme precision on the CPU.
        //    - However, GPUs are optimized for `Float` (32-bit). We cast the difference
        //      here so it can be packed into our `Uniforms` struct and sent to the
        //      Metal shaders.
        // 3. Motion Smoothing:
        //    - This value is the "gas pedal" for your liquid simulation. It ensures that
        //      the "gooey" flow remains fluid and consistent, even as the user
        //      scans through a list or triggers haptic feedback.
        let rawDelta = Float(currentTime - lastTime)
        // --- Closing the Time Loop (The Memory Handover) ---
        // Updates the persistent timestamp to prepare for the calculation of the next frame.
        // 1. Maintaining the Sequence:
        //    - Animation is simply a series of "Now" vs "Then" comparisons. By saving
        //      the current timestamp into `lastTime`, we establish the "Then" for
        //      the frame that will arrive in roughly 8 or 16 milliseconds.
        // 2. Preventing Accumulation Errors:
        //    - We perform this update *after* calculating the current frame's delta but
        //      *before* the function exits. This creates a seamless chain of time segments
        //      covering the entire duration the app is running.
        // 3. The State Anchor:
        //    - Because `lastTime` is a property of the class (self), it survives the
        //      end of this function call. This is what allows the `ParticleRenderer`
        //      to have a "memory" of the simulation's progress across thousands
        //      of individual draw calls.
        lastTime = currentTime
        // --- Calculating Simulation Age (The Relative Clock) ---
        // Converts the massive system timestamp into a local "Playhead" time.
        // 1. Starting from Zero:
        //    - By subtracting the `startTime` (captured at birth) from the `currentTime`,
        //       we create a value that starts at 0.0 and climbs upward.
        //    - This makes your shader math much simpler; for example, a sine wave
        //      like `sin(time)` will always start its first oscillation at the exact
        //      moment the view appears.
        // 2. Maintaining Floating-Point Precision:
        //    - System timestamps (seconds since boot) can become very large, which
        //      causes "jitter" in 32-bit GPU math due to precision loss.
        //    - Keeping the `time` value relative and small ensures that your
        //      liquid "wobbles" and particle fades remain buttery smooth even
        //      if the app has been running for hours.
        // 3. The Global Pulse:
        //    - This single value is passed to every shader, acting as the "Metronome"
        //      that synchronizes the movement of all 250 particles.
        self.time = Float(currentTime - startTime)
        
        // --- Implementing the Physics Safety Valve (The Delta Clamp) ---
        // Places an upper limit on how much time can "pass" in a single simulation step.
        // 1. Preventing the "Teleportation" Glitch:
        //    - If the app stutters (e.g., a heavy background task or a notification drop),
        //      'rawDelta' might jump from 0.016s to 0.5s or more.
        //    - Without this clamp, your particles would try to travel 30 frames worth of
        //      distance in a single leap, causing them to phase through boundaries or
        //      explode outward in a chaotic burst.
        // 2. Ensuring Simulation Stability:
        //    - By capping the delta at 0.1 seconds (100ms), we ensure that even during
        //      extreme lag, the physics engine treats the gap as a manageable step.
        //    - The animation might appear to "slow down" momentarily during a stutter,
        //      but it will remain mathematically stable and visually coherent.
        // 3. Balancing Realism and Reliability:
        //    - This is a standard practice in game development and high-end UI
        //      animations. It protects the integrity of your "liquid" math, ensuring the
        //      gooey particles never gain enough velocity to "break" the simulation.
        let dt = min(rawDelta, 0.1)
        
        // --- Capturing the Physics Driver (The Interaction Target) ---
        // Samples the current scroll position to use as the primary force for the simulation.
        // 1. Translating Gestures to Math:
        //    - The `scrollOffset` is a value passed in from the SwiftUI ScrollView.
        //    - By assigning it to `target`, we create a local snapshot that the GPU
        //      will use to pull particles downward. The further the user pulls, the
        //      higher this number, and the stronger the "gravity" acting on the liquid.
        // 2. The Anchor Point:
        //    - This value serves as the "Goal" for our particles. While they have their
        //      own velocity and wiggles, they are ultimately trying to follow this
        //      target. This is what makes the animation feel "attached" to the user's finger.
        // 3. Decoupling Input from Physics:
        //    - We capture this as a local constant before sending it to the GPU.
        //    - This ensures that even if the user moves their finger slightly *during* //      the frame calculation, the entire GPU workload for this specific frame
        //      is working toward a single, consistent coordinate.
        let target = self.scrollOffset
        // --- Calculating the Positional Gap (The Error Signal) ---
        // Determines the distance between where the animation currently is and where it needs to be.
        // 1. The Physics "Tug-of-War":
        //    - 'internalOffset' represents the "slow" version of the scroll—the one that
        //      has mass and momentum. 'target' is the "fast" raw scroll value from the finger.
        //    - The `difference` is essentially the tension in an invisible spring connecting
        //      the two. The larger this gap, the more force we will apply to "pull"
        //      the internal state toward the finger.
        // 2. Creating Organic "Lag":
        //    - We don't want the liquid to snap instantly to the finger (which looks stiff).
        //    - By calculating this difference, we can apply a fraction of it each frame,
        //      creating that smooth, high-end "Elastic" or "Gooey" delay that makes
        //      the pull-to-refresh feel premium.
        // 3. Directional Logic:
        //    - This value is signed. If it's positive, the user is pulling down further;
        //      if negative, the scroll is snapping back. This tells our particle engine
        //      whether to expand the "liquid" blob or compress it back into a ring.
        let difference = target - internalOffset
        
        /*
         // ==============================================================================
         // SCENARIO TRACE: Frame #6001
         // ------------------------------------------------------------------------------
         // • Context: The app has been running for exactly 100 seconds.
         // • User Action: The user is pulling down (Scroll Offset = 50.0).
         // • Current State: The heavy "liquid" is lagging behind at 30.0.
         // • Goal: Calculate the physics delta to bridge the gap between 30.0 and 50.0.
         // ==============================================================================
         
         // --- Step 1: Capturing the Present Moment (The Timestamp) ---
         // We freeze the system clock to ensure every calculation this frame uses the exact same time.
         // [Trace]: The system clock ticks forward. It is now 16ms later than the last frame.
         // [Value]: 100.016
         let currentTime = CACurrentMediaTime()
         
         // --- Step 2: The Bootstrap Check (The Safety Net) ---
         // If this is the very first frame (Frame #0), 'lastTime' would be 0.
         // Subtracting 100.016 - 0 would create a huge delta, exploding the physics.
         // [Trace]: 'lastTime' is 100.000 (set previous frame), so this check is skipped.
         if lastTime == 0 { lastTime = currentTime }
         
         // --- Step 3: Measuring the Interval (The Heartbeat) ---
         // We calculate exactly how much time has passed since we last drew the screen.
         // [Trace]: 100.016 (Now) - 100.000 (Then) = 0.016 seconds.
         // [Value]: 0.016
         let rawDelta = Float(currentTime - lastTime)
         
         // --- Step 4: The Time Clamp (The Speed Limit) ---
         // If the app stuttered and 'rawDelta' was 0.5s, the physics would jump too far.
         // We cap the maximum time step to 0.1s (100ms) to keep the simulation stable.
         // [Trace]: min(0.016, 0.1) -> 0.016. The frame was fast enough, no clamping needed.
         // [Value]: 0.016
         let dt = min(rawDelta, 0.1)
         
         // --- Step 5: Closing the Loop (The Handover) ---
         // We save the current time to be used as the "Previous Time" for the next frame.
         // [Trace]: 'lastTime' is updated from 100.000 to 100.016 for Frame #6002.
         lastTime = currentTime
         
         // --- Step 6: Simulation Age (The Global Clock) ---
         // We calculate the total time the animation has been alive. This drives the
         // sine waves and steady "wobble" of the liquid.
         // [Trace]: 100.016 (Now) - 0.000 (Start Time) = 100.016 seconds.
         // [Value]: 100.016
         self.time = Float(currentTime - startTime)
         
         // --- Step 7: Capturing the Target (The Goal) ---
         // We grab the user's actual finger position (or scroll position) from SwiftUI.
         // [Trace]: The user has pulled the list down by 50 points.
         // [Value]: 50.0
         let target = self.scrollOffset
         
         // --- Step 8: Calculating the Lag (The Tension) ---
         // We measure the distance between the user's finger (Target) and the
         // liquid's current heavy position (Internal Offset).
         // [Trace]: 50.0 (Finger) - 30.0 (Liquid) = +20.0.
         // [Logic]: The positive result means the liquid is "behind" and needs to be pulled down.
         // [Value]: 20.0
         let difference = target - internalOffset
         */
        
        // --- Detecting the Interaction Phase (The Directional Switch) ---
        // Determines whether the liquid is being actively pulled down or if it is snapping back up.
        // 1. The "Stretch" State (Positive Difference):
        //    - If 'difference' is greater than 0, the Target (Finger) is physically below
        //      the Liquid (Internal Offset).
        //    - This means the invisible spring connecting them is being STRETCHED. The user
        //      is pulling the refresh control down, adding tension to the system.
        // 2. Adjusting the Physics:
        //    - Inside this block, we will likely modify the particle behavior to simulate
        //      "Flow" or "Drip." When liquid is pulled, it tends to elongate and separate.
        //    - We might increase the effective gravity or chaos to make the pull feel
        //      heavy and viscous.
        // 3. The Alternative (Snapping Back):
        //    - If this were false (negative), the Liquid would be below the Target, implying
        //      the user released the scroll view and the spring is rapidly pulling the
        //      liquid back up to its resting position (Compression).
        if difference > 0 {

            // --- Normalizing the Force (The Magnitude) ---
            // Strips away the direction (positive/negative) to focus purely on the strength of the pull.
            // 1. Absolute Value:
            //    - The `difference` tells us *where* the pull is coming from (above or below),
            //      but for this specific calculation, we only care *how hard* it is pulling.
            //    - `abs()` converts both -20.0 (snapping back) and +20.0 (pulling down) into
            //      just 20.0.
            // 2. Physics Application:
            //    - We use this value to calculate the "Stretch" of our liquid.
            //    - Whether the liquid is flying up or down, the *amount* it distorts depends
            //      only on the speed/distance, not the direction.
            // 3. Preparation for Shaping:
            //    - This distance will determine how far the "Drop" hangs from the main ring.
            //      A larger distance means a longer, thinner droplet connection.
            let dist = abs(difference)
            
            // --- Creating the Transition Curve (The Interpolator) ---
            // Maps the raw distance (0 to ∞) into a normalized 0.0-to-1.0 percentage.
            // 1. The Smoothstep Function:
            //    - Unlike a linear map (which is a straight line), `smoothstep` creates an
            //      S-Curve (Hermite interpolation).
            //    - It starts slow, speeds up in the middle, and slows down at the end.
            //      This feels much more organic than a hard linear transition.
            // 2. The Range Definition:
            //    - Input 0.0 -> Output 0.0 (No stretch).
            //    - Input 100.0 -> Output 1.0 (Maximum stretch).
            //    - If `dist` is 50.0, the output isn't exactly 0.5; it's slightly different
            //      due to the curve, making the animation feel "easier" at the start.
            // 3. The Usage:
            //    - We will use this `blendFactor` to mix two different physics behaviors.
            //      At 0.0, the liquid is a tight circle. At 1.0, it is fully elongated.
            //      This value controls the "morph" between those two states.
            let blendFactor = smoothstep(0.0, 100.0, dist)

            // --- Calculating Dynamic Resistance (The Viscosity) ---
            // smoothly increases the "Drag" or "Heaviness" of the liquid as it stretches.
            // 1. The Linear Interpolation (Mix):
            //    - The `mix` function is a standard math operation: `start + (end - start) * factor`.
            //    - We start at 0.1 (Low Friction) and end at 0.4 (High Friction).
            //    - The `blendFactor` decides where we are between them.
            // 2. The Physics Simulation:
            //    - When the pull is small (blendFactor ≈ 0), the liquid is loose and watery (0.1).
            //    - As the user pulls further and the "goo" stretches out (blendFactor ≈ 1),
            //      we increase the friction to 0.4.
            // 3. The Tactile Result:
            //    - This simulates "Surface Tension." Just like stretching a real rubber band
            //      or taffy, it gets physically harder to move the further you pull it.
            //      This makes the UI feel "thicker" at the bottom of the pull.
            let pullFriction = mix(0.1, 0.4, blendFactor)
            
            // --- Frame-Rate Independent Damping (The Math Stabilizer) ---
            // Adjusts the friction value so the physics feel identical at 60Hz, 120Hz, or any fluctuating rate.
            // 1. The Problem (Time Slicing):
            //    - If we just applied `pullFriction` (e.g., 0.1) every frame, a device running at
            //      120fps would apply it twice as often as a 60fps device.
            //    - The animation would look "heavier" and sluggish on ProMotion screens because
            //      the friction would compound too quickly.
            // 2. The Solution (Exponential Decay):
            //    - We use the formula: `1 - (1 - friction) ^ (dt * targetFPS)`.
            //    - `dt * 60.0` normalizes the time step. If `dt` is exactly 1/60th of a second,
            //      the exponent is 1.0, and we get the raw friction.
            //    - If `dt` is smaller (1/120th), the exponent is 0.5, applying a mathematically
            //      smaller slice of friction that perfectly adds up over two frames.
            // 3. The Result:
            //    - This ensures your liquid simulation behaves consistently across all iPhones
            //      and iPads, regardless of their screen refresh rate or temporary lag spikes.
            let timeCorrected = 1.0 - pow(1.0 - pullFriction, dt * 60.0)
            
            // --- Applying the Weighted Follow (The Smooth Pursuit) ---
            // Moves the "heavy" liquid position a fraction of the distance toward the target.
            // 1. The Physics Calculation:
            //    - We take the gap between the liquid and the finger (`difference`) and multiply
            //      it by our calculated friction factor (`timeCorrected`).
            //    - If the gap is 100px and `timeCorrected` is 0.1, we move 10px closer.
            // 2. The Resulting Motion:
            //    - This is an "Exponential Smoothing" (or Low-Pass Filter).
            //    - It causes the liquid to accelerate quickly when the gap is large,
            //      but decelerate smoothly as it gets closer to the target, creating
            //      that signature "magnetic" or "heavy" feel.
            // 3. Updating the State:
            //    - By adding this result to `internalOffset`, we update the "Ghost" position
            //      for the next frame. The GPU will render the liquid at this new spot,
            //      slightly closer to the finger than it was 16ms ago.
            internalOffset += difference * timeCorrected
        } else {
            // RELEASING: (Keep your existing logic here)
            
            // --- Normalizing the Snap Distance (The Magnitude) ---
            // Converts the negative "Overshoot" into a positive distance value for physics calculations.
            // 1. The Directional Context:
            //    - Since we are in the `else` block, `difference` is negative (e.g., -20.0).
            //    - This means the liquid is currently *below* the target and needs to snap back *up*.
            // 2. The Absolute Necessity:
            //    - Physics formulas for friction and interpolation (like `smoothstep`) require
            //      positive numbers to function correctly. You cannot have a "negative distance,"
            //      only a negative direction.
            //    - `abs(-20.0)` gives us `20.0`, essentially saying "The liquid is 20 points away
            //      from home," regardless of which side it's on.
            // 3. The Usage:
            //    - This value becomes the input for our decay curve. It tells the system
            //      how much "Snap" energy is remaining. As this number approaches 0,
            //      the spring force gently fades out to prevent a harsh stop.
            let dist = abs(difference)
            
            // --- Defining the Snap Behavior Zones (The Velocity Map) ---
            // Creates a "Soft Landing" zone by mapping the distance into a physics control value.
            // 1. The "Dead Zone" (0 to 20.0):
            //    - The first parameter is 20.0. This means if the liquid is within 20 points
            //      of the target (very close), the output is strictly 0.0.
            //    - This ensures that the final few pixels of the "snap back" always use the
            //      exact same physics (the "Landing" friction), preventing the animation
            //      from feeling jittery or unstable right at the end.
            // 2. The "Acceleration Zone" (20.0 to 100.0):
            //    - As the distance grows from 20 to 100, the value transitions smoothly
            //      from 0.0 to 1.0.
            //    - This creates a gradient: "Far away = High Energy" vs "Close = Gentle Settlement."
            // 3. The Usage:
            //    - We will use this factor to interpolate our friction.
            //    - When far away (1.0), we want the liquid to snap back quickly.
            //    - When close (0.0), we want it to slow down to park itself softly without
            //      bouncing endlessly.
            let blendFactor = smoothstep(20.0, 100.0, dist)
            
            // --- Calculating the Snap Dynamics (The Damping Curve) ---
            // Interpolates the liquid's behavior between "Snappy" and "Heavy" based on distance.
            // 1. The Low Friction Zone (0.015):
            //    - When `blendFactor` is 0.0 (close to the target, < 20px), we use 0.015.
            //    - This is extremely low friction. It allows the liquid to oscillate or "wobble"
            //      freely as it settles into its final resting place, creating a lively
            //      jelly-like finish.
            // 2. The High Friction Zone (0.4):
            //    - When `blendFactor` is 1.0 (far away, > 100px), we use 0.4.
            //    - This is high friction. It prevents the liquid from flying back too violently
            //      when released from a deep pull, keeping the motion controlled and elegant.
            // 3. The Result:
            //    - As the liquid snaps back (distance decreases), the friction automatically
            //      drops from 0.4 down to 0.015.
            //    - It starts heavy (controlled) and ends loose (bouncy), perfectly mimicking
            //      the physics of a droplet hitting a surface.
            let baseFriction = mix(0.015, 0.4, blendFactor)
            
            // --- Frame-Rate Independent Damping (The Math Stabilizer) ---
            // Adjusts the friction value so the physics feel identical at 60Hz, 120Hz, or any fluctuating rate.
            // 1. The Problem (Time Slicing):
            //    - If we just applied `baseFriction` (e.g., 0.4) every frame, a device running at
            //      120fps would apply it twice as often as a 60fps device.
            //    - The animation would look "heavier" and sluggish on ProMotion screens because
            //      the friction would compound too quickly.
            // 2. The Solution (Exponential Decay):
            //    - We use the formula: `1 - (1 - friction) ^ (dt * targetFPS)`.
            //    - `dt * 60.0` normalizes the time step. If `dt` is exactly 1/60th of a second,
            //      the exponent is 1.0, and we get the raw friction.
            //    - If `dt` is smaller (1/120th), the exponent is 0.5, applying a mathematically
            //      smaller slice of friction that perfectly adds up over two frames.
            // 3. The Result:
            //    - This ensures your liquid simulation behaves consistently across all iPhones
            //      and iPads, regardless of their screen refresh rate or temporary lag spikes.
            let timeCorrectedFriction = 1.0 - pow(1.0 - baseFriction, dt * 60.0)
            
            // --- Applying the Restoring Force (The Spring Snap) ---
            // Moves the heavy liquid back toward its resting position using the calculated friction.
            // 1. The Physics Calculation:
            //    - We take the gap (`difference`, which is negative here) and multiply it by
            //      our adaptive friction (`timeCorrectedFriction`).
            //    - Since `difference` is negative, adding this result *subtracts* from
            //      `internalOffset`, effectively pulling the liquid UP toward 0.0.
            // 2. The Variable Speed:
            //    - Unlike a constant speed animation, this uses the "Variable Friction" we
            //      just calculated.
            //    - It snaps back quickly at first (high friction), then loosens up (low friction)
            //      near the top to allow for a gentle, organic settle.
            // 3. The Result:
            //    - This line executes the actual movement. It bridges the gap between the
            //      "Expanded" state and the "Resting" state, ensuring the liquid doesn't
            //      just teleport home but flows back naturally.
            internalOffset += difference * timeCorrectedFriction
            
            // --- Implementing the "Dead Zone" Snap (The Clean Finish) ---
            // prevents the infinite Zeno's Paradox of asymptotic movement.
            // 1. The Mathematical Problem:
            //    - Our physics formula (`offset += diff * friction`) moves the liquid *fractionally* //      closer every frame (e.g., 10.0 -> 5.0 -> 2.5 -> 1.25 -> 0.625...).
            //    - Mathematically, it never actually reaches 0.0; it just gets infinitely close,
            //      wasting CPU cycles calculating microscopic movements the user can't see.
            // 2. The Threshold Check:
            //    - We check if the liquid is within half a pixel (0.5) of its final destination.
            //    - This is visually indistinguishable from "Arrived" on high-density screens.
            // 3. The Hard Snap:
            //    - If close enough, we force `internalOffset` to exactly equal `target`.
            //    - This stops the physics engine instantly, ensuring the animation settles
            //      cleanly and the specialized "Wobble" shaders can take over
            //      without fighting residual drift.
            if dist < 0.5 { internalOffset = target }
        }

        //--- The Uniforms (The Global State) ---
        // This struct packages the "World Settings" to send from the CPU (Swift) to the GPU (Metal).
        // These values are constant across all particles for this specific frame.
        // 1. screenSize:
        //    - "The Canvas Dimensions."
        //    - We use `drawableSize` (actual pixels) rather than `frame.size` (logical points)
        //      to ensure crisp rendering on Retina/High-DPI displays.
        //    - The GPU uses this to calculate aspect ratios and normalize coordinates (0.0 to 1.0).
        // 2. scrollOffset:
        //    - "The Camera Position."
        //    - Represents how far the user has scrolled or moved the view.
        //    - In the Vertex Shader, we subtract this from the particle's world position
        //      to determine where it draws on the screen.
        // 3. time:
        //    - "The Heartbeat."
        //    - A continuous clock value used to drive animations, sinusoidal waves,
        //      or physics integrations that rely on absolute time.
        var uniforms = Uniforms(
            screenSize: SIMD2(Float(view.drawableSize.width), Float(view.drawableSize.height)),
            scrollOffset: internalOffset,
            time: time
        )
        
        //--- The Parameters (The Style & Constraints) ---
        // This struct bundles the "Tweakable Knobs" that define the look and limits of the system.
        // Unlike Uniforms (which are environment state), these are often values controlled by UI sliders.
        // 1. color:
        //    - "The Liquid Hue."
        //    - We convert the high-level Swift color (UIColor/NSColor) into a GPU-friendly
        //      SIMD vector (Red, Green, Blue, Alpha) normalized between 0.0 and 1.0.
        // 2. particleCount:
        //    - "The Census."
        //    - We cast the count to Int32 (the standard integer type for Metal shaders).
        //    - The GPU uses this for loop bounds or safety checks to ensure threads don't
        //      access invalid memory.
        // 3. triangleTop / triangleBottom:
        //    - "The Container Geometry."
        //    - These define specific coordinates/limits for the simulation shape (likely an hourglass or funnel).
        //    - The Physics Kernel uses these to calculate collisions or constrain movement.
        // 4. opacityInCircle:
        //    - "The Viscosity Visual."
        //    - Controls the alpha curve inside the particle's radial gradient.
        //    - High values = thick/solid paint; Low values = misty/gaseous look.
        // 5. glowIntensity:
        //    - "The Bloom Factor."
        //    - A multiplier applied to the final color.
        //    - Values > 1.0 allow the colors to exceed standard brightness, creating
        //      a glowing effect when combined with proper blending.
        var params = Parameters(
            color: getSIMDColor(from: self.color),
            particleCount: Int32(particleCount),
            triangleTop: triangleTop,
            triangleBottom: triangleBottom,
            opacityInCircle: opacityInCircle,
            glowIntensity: glowIntensity
        )

        // --- Initiating the Compute Pass (The GPU Workspace) ---
        // Creates a command encoder specifically designed for general-purpose parallel processing (GPGPU).
        // 1. The Context Switch:
        //    - Up until now, we've been running on the CPU (Swift). Now, we need to instruct
        //      the GPU to perform massive parallel calculations (updating 250 particle positions).
        //    - `makeComputeCommandEncoder` sets up the GPU pipeline for "Compute Kernels"
        //      instead of standard rendering (triangles/fragments).
        // 2. The Sandbox Creation:
        //    - This encoder acts as a recording tape. We will record a series of commands
        //      (set buffer, set texture, dispatch threads) onto it.
        //    - Nothing executes yet; we are simply building the "ToDo List" for the GPU.
        // 3. Safety First:
        //    - This returns an Optional because the GPU might be busy or the command buffer
        //      invalid. The `if let` ensures we only proceed if the hardware is ready to
        //      accept commands, preventing crashes.
        if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
            // --- Loading the Brain (The Shader Program) ---
            // Tells the GPU exactly which function (Kernel) to execute for this pass.
            // 1. The "Cartridge" Analogy:
            //    - The GPU is a blank slate. It doesn't know what "particles" or "liquid" are.
            //    - `computeState` acts like a game cartridge. It contains the pre-compiled
            //      binary code of our specific Metal function (likely named `particleUpdate`).
            // 2. The Context Switch:
            //    - By setting this state, we lock the GPU into "Physics Mode."
            //    - Any subsequent dispatch commands will use the logic inside that specific
            //      shader function to process data.
            // 3. Performance Note:
            //    - Creating this state object (compiling the .metal file) is slow and happened
            //      at app launch.
            //    - Switching to it here is incredibly fast—it's just a pointer swap that
            //      takes mere nanoseconds.
            computeEncoder.setComputePipelineState(computeState)
            // --- Binding the Particle Memory (The Data Feed) ---
            // Connects our CPU-managed list of particles to the GPU's "Slot 0."
            // 1. The Physical Connection:
            //    - `particleBuffer` is the massive array in memory that holds the position (x,y)
            //      and velocity (vx,vy) for every single dot in our animation.
            //    - By setting it here, we give the GPU read/write access to this memory block.
            // 2. The Shader Slot (Index 0):
            //    - In the .metal file, the function signature looks like:
            //      `kernel void particleUpdate(device Particle *particles [[buffer(0)]], ...)`
            //    - The `index: 0` in Swift must match the `[[buffer(0)]]` in Metal perfectly.
            //      This is how the code knows which variable is which.
            // 3. The Persistent State:
            //    - This buffer is special. The GPU reads the *current* positions from it,
            //      calculates the physics, and writes the *new* positions back into the
            //      same spot.
            //    - This allows the simulation to have "Memory" of where particles were
            //      in the previous frame.
            computeEncoder.setBuffer(particleBuffer, offset: 0, index: 0)
            // --- Broadcasting the Global State (The Constant Feed) ---
            // Injects our "Care Package" of variables directly into the GPU's high-speed constant memory.
            // 1. Direct Injection (setBytes vs setBuffer):
            //    - Unlike the particle array (which is massive and stays in memory), our `uniforms`
            //      struct is tiny (< 4KB).
            //    - `setBytes` copies this data *directly* into the command buffer itself. It avoids
            //      the overhead of allocating a new GPU buffer for every single frame, making it
            //      extremely fast for changing variables like Time or Touch Position.
            // 2. The Synchronization Point:
            //    - This ensures that every single one of the 250 threads (particles) reads the
            //      exact same `time` (100.016) and `scrollOffset` (32.0) for this frame.
            //    - Without this, particles might drift out of sync or use outdated physics.
            // 3. The Memory Layout (Stride):
            //    - We use `.stride` (not `.size`) because it includes the necessary padding bytes
            //      to satisfy the GPU's 16-byte alignment requirement.
            //    - This prevents the GPU from reading "half a float" and corrupting the simulation.
            computeEncoder.setBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 1)
            //--- The Direct Upload (Data Injection) ---
            // This command sends the "Parameters" struct from CPU memory directly to the GPU.
            // We use `setBytes` (instead of creating a dedicated MTLBuffer) because this data is
            // small (less than 4KB) and changes every frame, making this method faster and simpler.
            // 1. &params:
            //    - The "Payload."
            //    - We pass the memory address (pointer) of our Swift struct.
            // 2. length: MemoryLayout<Parameters>.stride:
            //    - The "Payload Size."
            //    - We use `.stride` instead of `.size` to account for any memory padding
            //      required by the alignment rules. This ensures the GPU reads the full,
            //      correctly aligned block of memory.
            // 3. index: 2:
            //    - The "Mailbox Number."
            //    - This MUST match the `[[buffer(2)]]` attribute in your Metal shader signature.
            //    - If these numbers don't match, the shader will look in the wrong slot
            //      and likely read garbage data (zeros or random noise).
            computeEncoder.setBytes(&params, length: MemoryLayout<Parameters>.stride, index: 2)
            
            // --- Querying the Hardware Architecture (The SIMD Lane Width) ---
            // Determines the "Natural Batch Size" of your specific iPhone's GPU hardware.
            // 1. Parallel Processing "Buckets":
            //    - A GPU doesn't run one task at a time; it runs groups of tasks in lockstep
            //      (called a "Warp" or "Wavefront").
            //    - `executionWidth` tells us how many threads the GPU executes
            //      simultaneously in a single clock cycle. On most modern iPhones, this
            //      value is 32.
            // 2. Optimization Strategy:
            //    - By capturing this value, we can ensure our "Thread Groups" are a
            //      perfect multiple of the hardware's capacity.
            //    - If we processed 31 particles instead of 32, the GPU would still
            //      spend the energy for 32, leaving one "lane" idle and wasting performance.
            // 3. Hardware Independence:
            //    - Different chips (A15 vs M4) might have different widths. By querying
            //      this dynamically, we ensure the liquid physics engine runs at peak
            //      efficiency on any Apple device.
            let executionWidth = computeState.threadExecutionWidth
            // --- Defining the Work Unit (The Threadgroup Dimensions) ---
            // Organizes individual threads into a cohesive block for the GPU to process.
            // 1. The Threadgroup Concept:
            //    - GPU work is divided into "Threadgroups." Threads within the same group
            //      can share memory and synchronize with each other.
            //    - Think of this as defining the "Size of the Bus" that carries our
            //      particles to the processor.
            // 2. Linear Organization (1D Grid):
            //    - Since our particles are stored in a simple flat array (0 to 249),
            //      we use a width of `executionWidth` (32) and a height/depth of 1.
            //    - This creates a thin, 1D "strip" of work that perfectly matches the
            //      hardware's execution lanes.
            // 3. Efficiency:
            //    - By setting the width to exactly `executionWidth`, we ensure that every
            //      "seat" on the GPU's processing bus is filled, maximizing throughput
            //      for our liquid simulation.
            let threadsPerGroup = MTLSize(width: executionWidth, height: 1, depth: 1)
            // --- Calculating the Dispatch Grid (The Integer Ceiling Math) ---
            // Determines exactly how many threadgroups are needed to cover every particle.
            // 1. The "Leftover" Problem:
            //    - If we have 250 particles and an execution width of 32, a simple
            //      division (250 / 32) equals 7.81.
            //    - Since we can't launch "0.81" of a group, we must round UP to 8 groups
            //      to ensure particles 225 through 250 aren't ignored.
            // 2. The Integer Division Trick:
            //    - In Swift, `250 / 32` would return 7 (truncating the decimal).
            //    - The formula `(count + width - 1) / width` is a standard GPU pattern
            //      to perform a "Ceiling" operation using only integers.
            //    - (250 + 32 - 1) / 32  =>  281 / 32  =>  8.78... which becomes 8.
            // 3. The Result:
            //    - This ensures we launch enough "buses" (threadgroups) to pick up every
            //      single particle in the simulation, even if the last bus is partially empty.
            let groupsWidth = (particleCount + executionWidth - 1) / executionWidth
            // --- Finalizing the Dispatch Map (The Grid Blueprint) ---
            // Defines the total number of work units (groups) we are sending to the GPU.
            // 1. The Full Assembly:
            //    - If `threadsPerGroup` defined the size of a single "bus," then
            //      `threadgroupsPerGrid` defines how many of those buses we are
            //      sending out into the city (the GPU).
            // 2. Mapping the Counts:
            //    - We take our calculated `groupsWidth` (e.g., 8) and set it as the width.
            //    - Since our simulation is a 1D list of particles, we keep height and
            //      depth at 1. This creates a linear "convoy" of processing power.
            // 3. The Execution Logic:
            //    - By multiplying `threadgroupsPerGrid` by `threadsPerGroup`, the GPU
            //      knows the total "Threads per Grid" (8 * 32 = 256).
            //    - This ensures all 250 of our particles are covered, with 6 extra
            //      inactive threads acting as a safety buffer at the end of the line.
            let threadgroupsPerGrid = MTLSize(width: groupsWidth, height: 1, depth: 1)
            // --- Launching the Simulation (The GPU Ignition) ---
            // Pulls the trigger and tells the GPU to begin executing the recorded commands in parallel.
            // 1. The Command Handover:
            //    - This is the moment the "Instructions" (the kernel) meet the "Data" (the particles).
            //    - Metal uses the `threadgroupsPerGrid` blueprint to spawn the exact number of
            //      parallel workers needed to update the entire liquid simulation at once.
            // 2. Parallel Explosion:
            //    - Instead of the CPU updating particles 1, 2, 3... in a slow loop, the GPU
            //      starts all 250+ updates simultaneously.
            //    - Each particle calculates its own new position based on the `internalOffset`
            //      and `time` we passed in earlier.
            // 3. Asynchronous Nature:
            //    - This call is non-blocking. The CPU doesn't wait for the GPU to finish; it
            //      simply hands off the work order and immediately moves to the next line of
            //      code, allowing your app to stay responsive even during heavy physics.
            computeEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerGroup)
            // --- Sealing the Work Order (The Encoding Wrap-Up) ---
            // Finalizes the list of commands for this specific compute pass.
            // 1. Closing the Encoder:
            //    - Think of the `computeEncoder` as a recording session. By calling
            //      `endEncoding()`, we are saying "The recording is finished; no more
            //      commands will be added to this pass."
            // 2. Resource Release:
            //    - This releases the encoder's hold on the GPU resources (like the
            //      particle buffer and pipeline state). It prepares the `commandBuffer`
            //      so that other encoders (like a Render Encoder for drawing) can
            //      take over the hardware.
            // 3. Preparation for Submission:
            //    - You cannot submit a command buffer to the GPU until all its encoders
            //      have been properly closed. This line is the "handshake" that
            //      confirms the physics update is ready for the next stage of the pipeline.
            computeEncoder.endEncoding()
        }
        
        // --- Initiating the Render Pass (The Artist's Canvas) ---
        // Creates an encoder that translates our data into visible pixels on the screen.
        // 1. The Context Shift:
        //    - We just finished the "Compute" phase (math). Now we enter the "Render" phase (drawing).
        //    - The `renderPassDescriptor` contains the setup for the target texture (the screen),
        //      defining things like the background "clear" color and how to handle anti-aliasing.
        // 2. Setting the Destination:
        //    - This encoder is bound to the current frame's drawable. Any `draw` commands
        //      issued here will directly affect what the user sees on their iPhone display.
        // 3. The Graphics Pipeline:
        //    - Unlike the Compute Encoder which just runs logic, the Render Encoder
        //      orchestrates the Vertex Shader (positioning shapes) and the Fragment Shader
        //      (coloring pixels).
        // 4. Safety Check:
        //    - The `if let` ensures that a valid "canvas" exists. If the view is off-screen
        //      or the system is out of memory, we skip the drawing work to save power.
        if let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) {
            // --- Loading the Visual Style (The Graphics Pipeline) ---
            // Configures the GPU with the specific shaders and rules for drawing the liquid.
            // 1. The Fragment & Vertex Bond:
            //    - `renderState` is a pre-compiled object that links our Vertex Shader
            //      (which positions the triangles) and our Fragment Shader (which creates
            //      the "metaball" gooey effect).
            // 2. The Rendering Rules:
            //    - Beyond just the code, this state tells the GPU how to handle "Blending"
            //      (how the liquid colors mix with the background) and "Rasterization"
            //      (how to turn mathematical shapes into physical screen pixels).
            // 3. Efficiency Check:
            //    - Like the compute state, this was compiled when the app first loaded.
            //      Setting it here is a "Zero-Cost" operation that instantly prepares
            //      the hardware for the complex math required to render the
            //      liquid-stretching effect.
            renderEncoder.setRenderPipelineState(renderState)
            // --- Feeding the Vertex Data (The Geometry Source) ---
            // Transfers the updated particle positions from the Compute pass into the Rendering pass.
            // 1. The Data Handover:
            //    - `particleBuffer` now contains the fresh (x, y) coordinates calculated
            //      by the GPU just microseconds ago in the compute encoder.
            //    - By setting it as a Vertex Buffer, we tell the GPU: "Use these points as
            //      the foundation for the shapes you are about to draw."
            // 2. Vertex Shader Access (Index 0):
            //    - In your .metal file, the Vertex Shader expects data at `[[buffer(0)]]`.
            //    - The GPU will loop through this buffer, treating each particle as a
            //      vertex that can be expanded into a liquid "blob."
            // 3. Memory Re-use (Zero-Copy):
            //    - This is the "Secret Sauce" of Metal. We aren't copying data back to
            //      the CPU and then back to the GPU.
            //    - The data stays exactly where it is in VRAM; we are simply telling the
            //      graphics hardware to look at the same memory address from a different
            //      perspective (Rendering instead of Computing).
            renderEncoder.setVertexBuffer(particleBuffer, offset: 0, index: 0)
            // --- Syncing the Global State (The Shader Constants) ---
            // Passes the layout and environmental data to the Vertex Shader at "Slot 1."
            // 1. Sharing the Logic:
            //    - Just as the Compute Shader needed to know the `time` and `scrollOffset`
            //      to calculate physics, the Vertex Shader needs this same data to
            //      position the liquid correctly on your screen.
            // 2. Small Data Optimization (setVertexBytes):
            //    - Because our `uniforms` struct is small, we use `setVertexBytes`.
            //    - This is faster than creating a formal `MTLBuffer` because it copies
            //       the data directly into the GPU's command stream, reducing the
            //       "overhead" (the administrative cost) of the draw call.
            // 3. Stride & Alignment:
            //    - We use `MemoryLayout<Uniforms>.stride` to ensure we include any
            //      necessary padding bytes. This ensures the GPU reads the struct
            //      data in 16-byte chunks, preventing "memory misalignment" which
            //      could cause the liquid to flicker or glitch.
            renderEncoder.setVertexBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 1)
            //--- The Vertex Configuration (Shader Inputs) ---
            // This command replicates the data upload, but specifically for the *Vertex Shader*.
            // While the Compute Kernel needed these params for physics, the Vertex Shader needs
            // them to determine how to draw the points (e.g., calculating point size or position).
            // 1. setVertexBytes:
            //    - We use the "Bytes" method again for speed and low overhead.
            //    - This bypasses the need for a dedicated buffer allocation for this frame,
            //      copying the data directly into the command buffer.
            // 2. index: 2:
            //    - The "Binding Slot."
            //    - This MUST match `constant Parameters &parameters [[buffer(2)]]` in your
            //      Vertex Shader function signature.
            //    - Note: Even if you sent this data to the Compute shader, you must send it
            //      again here if the Vertex shader needs to read it.
            renderEncoder.setVertexBytes(&params, length: MemoryLayout<Parameters>.stride, index: 2)
            //--- The Fragment Configuration (The Paint Supply) ---
            // This command delivers the "Parameters" struct to the final stage of the pipeline:
            // the Fragment Shader (The Painter).
            // 1. setFragmentBytes:
            //    - Unlike the Vertex stage (which handles geometry/position), this stage handles
            //      *color* and *pixels*.
            //    - We must upload the data *again* here because the Fragment Shader runs
            //      separately and has its own distinct memory slots.
            // 2. index: 2:
            //    - The "Binding Slot."
            //    - This MUST match `constant Parameters &parameters [[buffer(2)]]` in your
            //      Fragment Shader function signature.
            //    - Without this, the shader wouldn't know the base color, glow intensity,
            //      or alpha thresholds needed to render the final liquid look.
            renderEncoder.setFragmentBytes(&params, length: MemoryLayout<Parameters>.stride, index: 2)
            // --- Executing the Draw Call (The Final Painting) ---
            // Commands the GPU to actually process the vertices and turn them into pixels.
            // 1. The Primitive Type (.point):
            //    - We are telling Metal to treat every entry in our particle buffer as a
            //      single coordinate in 2D space (a "Point").
            //    - In the shader, we can then expand these single points into larger
            //      circular "blobs" using the `[[point_size]]` attribute.
            // 2. The Range (vertexStart to vertexCount):
            //    - `vertexStart: 0`: Begin drawing from the very first particle in our buffer.
            //    - `vertexCount: particleCount`: Draw exactly 250 particles.
            //    - This loop happens entirely on the GPU hardware; it is thousands of times
            //      faster than a `for` loop in Swift.
            // 3. The Resulting Magic:
            //    - This single line of code triggers the Vertex Shader to position the dots
            //      and the Fragment Shader to color them.
            //    - This is where our math finally transforms into the "gooey" liquid
            //      visuals the user sees on the screen.
            renderEncoder.drawPrimitives(type: .point, vertexStart: 0, vertexCount: particleCount)
            // --- Finalizing the Visual Pass (The Canvas Cleanup) ---
            // Closes the render command encoder and locks in the drawing instructions.
            // 1. Completing the "Paint Job":
            //    - Just as we did with the Compute Encoder, we must formally end the
            //      Rendering phase. This tells Metal we are finished submitting triangles,
            //      points, or textures for this specific frame.
            // 2. State Reset:
            //    - Calling this function clears the internal state of the encoder,
            //      letting the Command Buffer know that the "Render" portion of the
            //      GPU pipeline is ready to be bundled up.
            // 3. Thread Safety:
            //    - Once `endEncoding` is called, you can no longer send drawing commands
            //      to this encoder. It ensures that the GPU receives a complete,
            //      uninterrupted sequence of instructions, preventing partial or
            //      glitched frames from being sent to the display.
            renderEncoder.endEncoding()
        }
        
        // --- Scheduling the Reveal (The Display Handshake) ---
        // Registers the completed frame to be shown on the physical screen.
        // 1. The "Flip" Mechanism:
        //    - To avoid flickering, Metal uses "Double Buffering." While we were
        //      drawing on this `drawable`, the screen was showing the *previous* frame.
        //    - `present()` tells the system: "I am done painting. As soon as the screen
        //      refreshes (VSync), swap the old image for this new one."
        // 2. Synchronized Timing:
        //    - This ensures the liquid animation only updates when the screen is ready
        //      to display it (usually 60Hz or 120Hz on ProMotion displays).
        //    - It prevents "Screen Tearing," where the top half of the screen shows
        //      one frame and the bottom half shows another.
        // 3. Efficiency & Battery:
        //    - By scheduling the presentation here, we allow the OS to manage power
        //      consumption effectively, only waking up the display hardware when
        //      there is a fresh, completed frame ready to go.
        commandBuffer.present(drawable)
        // --- Executing the Command Queue (The "Go" Signal) ---
        // Submits the entire bundle of recorded instructions to the GPU hardware.
        // 1. The Final Handover:
        //    - Up until this point, we’ve just been "recording" a list of chores into
        //      the `commandBuffer`. Nothing has actually happened on the GPU yet.
        //    - `commit()` ships the entire "package" (Compute physics + Render pixels)
        //      to the GPU's command queue for immediate execution.
        // 2. Non-Blocking Execution:
        //    - This is the beauty of Metal: the CPU doesn't sit around waiting for the
        //      GPU to finish drawing. It drops the "work order" off and immediately
        //      moves on to handle the next frame or user touch input.
        // 3. Closing the Loop:
        //    - This line marks the official end of the frame's lifecycle. Within a few
        //      milliseconds, the GPU will chew through these instructions, and the
        //      user will see their gooey liquid animation move perfectly across the screen.
        commandBuffer.commit()
    }
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}
    
    // --- The Linear Interpolation Utility (The "Mix" Function) ---
    // A fundamental graphics math tool used to smoothly transition between two values.
    // 1. The GLSL/Metal Equivalent:
    //    - In Shaders, this is a built-in hardware-accelerated function. Since we are
    //      performing physics calculations on the CPU in Swift, we recreate it here
    //      to maintain mathematical consistency between our CPU and GPU logic.
    // 2. How the Math Works:
    //    - It calculates: a + (b - a) * t
    //    - If `t` is 0.0, it returns `a`.
    //    - If `t` is 1.0, it returns `b`.
    //    - If `t` is 0.5, it returns the perfect midpoint.
    // 3. Use Case in Liquid Physics:
    //    - We use this to "blend" values together, such as transitioning the color of
    //      the liquid during a pull or smoothing out the velocity of particles so
    //      they don't look jittery.
    func mix(_ a: Float, _ b: Float, _ t: Float) -> Float {
        return a + (b - a) * t
    }
    
    // --- The Organic Interpolation Utility (The "Smoothstep" Function) ---
    // A mathematical "Ease-In/Ease-In" function used to create smooth, curved transitions.
    // 1. Beyond Linear:
    //    - Unlike `mix` (which is a straight line), `smoothstep` produces an "S-curve."
    //    - It starts slow, speeds up in the middle, and slows down again at the end.
    //    - This is what gives the liquid its "heavy" and "viscous" feel rather than
    //      looking like rigid plastic.
    // 2. The Clamping Logic:
    //    - First, it clamps `x` between `edge0` and `edge1` to find a normalized value (0 to 1).
    //    - Then, it applies the cubic Hermite formula: 3x² - 2x³.
    // 3. Why We Use It Here:
    //    - In our pull-to-refresh, we use this to define the "threshold" of the gooey
    //      stretching. It ensures that the "stringy" parts of the liquid don't just
    //      snap into existence, but grow and shrink with a natural, elastic acceleration.
    func smoothstep(_ edge0: Float, _ edge1: Float, _ x: Float) -> Float {
        // --- Step 1: Normalization & Boundary Clamping ---
        // Converts the raw input value 'x' into a standardized 0.0 to 1.0 range.
        // 1. The Normalization Math:
        //    - By calculating `(x - edge0) / (edge1 - edge0)`, we determine where `x`
        //      sits relative to our two boundaries.
        //    - If `x` is exactly at `edge0`, the result is 0.0. If it's at `edge1`,
        //      the result is 1.0.
        // 2. The `simd_clamp` Safety Net:
        //    - In graphics, we often deal with values that overshoot their targets.
        //    - This ensures that even if `x` is way beyond our edges, the result
        //      never drops below 0.0 or exceeds 1.0. This prevents the "S-curve"
        //      math in the next step from breaking or producing inverted colors.
        // 3. Visual Consistency:
        //    - This is the "Linear" foundation. We define the start and end of the
        //      liquid's movement before we apply the "Smooth" easing logic.
        let t = simd_clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
        // --- Step 2: Applying the Cubic Hermite Curve (The "S-Curve") ---
        // Smooths the linear transition into an elegant, non-linear acceleration.
        // 1. The Power of the Polynomial:
        //    - The formula `3t² - 2t³` is the classic "Hermite" interpolation.
        //    - By squaring and cubing our normalized value `t`, we create a curve
        //      that is flat at the start (0.0) and flat at the end (1.0).
        // 2. Erasing the "Mechanical" Look:
        //    - Linear movement (a straight line) looks like a machine—it starts
        //      and stops instantly. This cubic curve creates a natural "Ease-In"
        //      and "Ease-Out."
        // 3. Why it matters for "Gooey" Effects:
        //    - When the liquid "stretches" or "breaks" during your pull-to-refresh,
        //      this math ensures the thickness changes subtly at the edges,
        //      making it look like real viscous fluid rather than a rigid rubber band.
        return t * t * (3.0 - 2.0 * t)
    }
}
