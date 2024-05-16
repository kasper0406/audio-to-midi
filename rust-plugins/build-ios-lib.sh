#!/bin/bash

set -e

# Build for all targets
cargo build --release --target aarch64-apple-ios-sim --features cbinds
cargo build --release --target x86_64-apple-ios --features cbinds
# cargo build --release --target x86_64-apple-darwin --features cbinds
# cargo build --release --target aarch64-apple-darwin --features cbinds

# Create a universal binary for iOS
mkdir -p target/universal-ios/release
IOS_OUTPUT="target/universal-ios/release/libmodelutil.dylib"
lipo -create -output "$IOS_OUTPUT" \
  target/aarch64-apple-ios-sim/release/libmodelutil.dylib \
  target/x86_64-apple-ios/release/libmodelutil.dylib

# Create a universal binary for macOS
#mkdir -p target/universal-darwin/release
#OSX_OUTPUT="target/universal-darwin/release/libmodelutil.dylib"
#lipo -create -output "$OSX_OUTPUT" \
#  target/x86_64-apple-darwin/release/libmodelutil.dylib \
#  target/aarch64-apple-darwin/release/libmodelutil.dylib

echo "iOS lib: $IOS_OUTPUT"
echo "OS X lib: $OSX_OUTPUT"

IOS_FRAMEWORK=target/universal-ios/release/ModelUtil.framework
rm -rf "$IOS_FRAMEWORK"
mkdir -p "$IOS_FRAMEWORK"
lipo -create "$IOS_OUTPUT" -output "$IOS_FRAMEWORK/ModelUtil"

# Copy in generated header files
cbindgen --config cbindgen.toml --output "target/universal-ios/release/Headers/ModelUtil.h"

tee "${IOS_FRAMEWORK}/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleIdentifier</key>
    <string>com.knielsen.audio2midi.ModelUtil</string>

    <key>CFBundleName</key>
    <string>ModelUtil</string>

    <key>CFBundleShortVersionString</key>
    <string>1.0.0</string>

    <key>CFBundleVersion</key>
    <string>1</string>

    <key>LSMinimumSystemVersion</key>
    <string>17.0</string>

    <key>CFBundlePackageType</key>
    <string>FMWK</string>

    <key>NSPrincipalClass</key>
    <string></string>

    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>

    <key>CFBundleSupportedPlatforms</key>
    <array>
        <string>iPhoneOS</string>
    </array>

    <key>UIRequiredDeviceCapabilities</key>
    <array>
        <string>arm64</string>
    </array>

    <key>CFBundleExecutable</key>
    <string>ModelUtil</string>
</dict>
</plist>
EOF

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

  s.vendored_frameworks     = 'ModelUtil.framework'
  # s.vendored_libraries      = 'ModelUtil.framework/libmodelutil.dylib'
  s.source_files            = 'Headers/*.h'
  s.public_header_files     = 'Headers/*.h'
end
EOF

echo "iOS Framework: ${IOS_FRAMEWORK}"
