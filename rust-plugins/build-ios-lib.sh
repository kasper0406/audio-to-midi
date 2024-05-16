#!/bin/bash

set -e

# Build for all targets
cargo build --release --target aarch64-apple-ios
cargo build --release --target x86_64-apple-ios
cargo build --release --target x86_64-apple-darwin
cargo build --release --target aarch64-apple-darwin

# Create a universal binary for iOS
mkdir -p target/universal-ios/release
IOS_OUTPUT="target/universal-ios/release/libmodelutil.dylib"
lipo -create -output $IOS_OUTPUT \
  target/aarch64-apple-ios/release/libmodelutil.dylib \
  target/x86_64-apple-ios/release/libmodelutil.dylib

# Create a universal binary for macOS
mkdir -p target/universal-darwin/release
OSX_OUTPUT="target/universal-darwin/release/libmodelutil.dylib"
lipo -create -output $OSX_OUTPUT \
  target/x86_64-apple-darwin/release/libmodelutil.dylib \
  target/aarch64-apple-darwin/release/libmodelutil.dylib

echo "iOS lib: $IOS_OUTPUT"
echo "OS X lib: $OSX_OUTPUT"
