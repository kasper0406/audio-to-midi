#!/bin/bash

set -e

# Build for all targets
cargo build --release --target aarch64-apple-ios-sim --features cbinds
cargo build --release --target aarch64-apple-ios --features cbinds
cargo build --release --target x86_64-apple-ios --features cbinds

# Create a universal binary for iOS
mkdir -p target/universal-ios/release

IOS_FRAMEWORK=target/universal-ios/release/ModelUtil.xcframework
rm -rf "$IOS_FRAMEWORK"
mkdir -p "$IOS_FRAMEWORK"

HEADERS="target/universal-ios/release/Headers"

# Create a fat library for the simulator as xcodebuild -create-xcframework fails to do it
lipo -create \
    target/aarch64-apple-ios-sim/release/libmodelutil.a \
    target/x86_64-apple-ios/release/libmodelutil.a \
    -output target/universal-ios/release/libmodelutil.a

# Copy in generated header files
cbindgen --config cbindgen.toml --output "$HEADERS/ModelUtil.h"

xcodebuild -create-xcframework \
    -library target/aarch64-apple-ios/release/libmodelutil.a -headers $HEADERS \
    -library target/universal-ios/release/libmodelutil.a -headers $HEADERS \
    -output "$IOS_FRAMEWORK"

tee "${IOS_FRAMEWORK}/../ModelUtil.podspec" <<EOF
Pod::Spec.new do |s|
  s.name             = 'ModelUtil'
  s.version          = '0.0.1'
  s.summary          = 'A simple library for interacting with the audio2midi model'
  s.homepage         = 'https://github.com/kasper0406/audio-to-midi'
  s.author           = { 'Kasper Nielsen' => 'kasper0406@gmail.com' }
  s.license          = { :type => 'MIT', :text => "Copyright 2024" }
  s.source           = { :http => 'https://github.com/kasper0406/audio-to-midi/rust-plugins' }
  s.platform         = :ios, '17.0'
  s.requires_arc     = true

  s.vendored_frameworks     = 'ModelUtil.xcframework'
  s.source_files            = 'Headers/*.h'
  s.public_header_files     = 'Headers/*.h'
end
EOF

echo "iOS Framework: ${IOS_FRAMEWORK}"
