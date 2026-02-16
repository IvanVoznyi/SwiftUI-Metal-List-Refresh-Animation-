//
//  ContentView.swift
//  Pull Refresh Animation
//
//  Created by Ivan Voznyi on 2/8/26.
//

import SwiftUI

enum Theme {
    case light
    case dark
}

struct ContentView: View {
    @State private var pullOffset: CGFloat = 0
    @State private var isTouching: Bool = false
    
    let thresholdRefresh: CGFloat = 75
    
    var backgroundColor: Color
    var particlesColor: Color
    var theme: Theme = .light
    var opacityInCircle: Float
    var glowIntensity: Float
    var triangleTop: Float
    var triangleBottom: Float

    init(
        backgroundColor: Color = .white,
        particlesColor: Color = .black,
        theme: Theme = .light,
        triangleTop: Float = 1.2,
        triangleBottom: Float = 0.25
    ) {
        self.theme = theme
        self.backgroundColor = backgroundColor
        self.particlesColor = particlesColor
        self.triangleTop = triangleTop
        self.triangleBottom = triangleBottom
        self.opacityInCircle = theme == .light ? 0.8 : 0.3
        self.glowIntensity = theme == .light ? 0.9 : 0.6
    }
    
    var body: some View {
        ZStack(alignment: .top) {
            Color(backgroundColor).opacity(1).ignoresSafeArea()
            
            MetalParticleView(
                scrollOffset: pullOffset,
                color: particlesColor,
                triangleTop: triangleTop,
                triangleBottom: triangleBottom,
                opacityInCircle: opacityInCircle,
                glowIntensity: glowIntensity
            )
            .frame(maxWidth: .infinity)
            .frame(height: 200)
            .zIndex(1)
            .allowsHitTesting(false)
            

            ScrollContent(pullOffset: $pullOffset)
                .simultaneousGesture(
                    DragGesture(minimumDistance: 0)
                        .onEnded { value in
                            if (pullOffset >= thresholdRefresh) {
                                print("refresh")
                            }
                        }
                )
        }
        .ignoresSafeArea()
        .onPreferenceChange(ScrollPreferenceKey.self) { currentPhysicsValue in
            pullOffset = min(max(0, currentPhysicsValue), thresholdRefresh)
        }
    }
}

struct ScrollPreferenceKey: PreferenceKey {
    static var defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value += nextValue()
    }
}

struct ScrollContent: View {
    @Binding var pullOffset: CGFloat

    var body: some View {
        ScrollView {
            VStack(spacing: 10) {
                // Add a spacer at top so particles aren't covered immediately
                Color.clear.frame(height: 100)
                ForEach(0..<50) { index in
                    Text("Item \(index)")
                        .frame(width: 200, height: 100)
                        .background(Color.white)
                        .foregroundStyle(.black)
                        .clipShape(RoundedRectangle(cornerRadius: 10))
                        .overlay(
                            RoundedRectangle(cornerRadius: 10)
                                .stroke(.black, lineWidth: 1)
                        )
                }
            }

            .padding()
            .background(
                GeometryReader { geometry in
                    Color.clear.preference(
                        key: ScrollPreferenceKey.self,
                        value: geometry.frame(in: .global).minY
                    )
                }
            )
        }
    }
}

#Preview("White Theme") {
    ContentView()
}


#Preview("Black Theme Straight Vertical Pink Particles") {
    ContentView(
        backgroundColor: .black,
        particlesColor: .pink,
        theme: .dark,
        triangleTop: 0.3,
        triangleBottom: 0.0
    )
}

#Preview("Black Theme") {
    ContentView(
        backgroundColor: .black,
        particlesColor: .cyan,
        theme: .dark
    )
}

#Preview("Black Theme Green particles") {
    ContentView(
        backgroundColor: .black,
        particlesColor: .green,
        theme: .dark
    )
}
