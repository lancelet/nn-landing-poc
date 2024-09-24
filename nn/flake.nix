{
  inputs = {
    naersk.url = "github:nix-community/naersk/master";
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    utils,
    naersk,
  }:
    utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {inherit system;};
        naersk-lib = pkgs.callPackage naersk {};
        isMacOS = system == "x86_64-darwin" || system == "aarch64-darwin";
        isLinux = system == "x86_64-linux" || system == "aarch64-linux" || system == "i686-linux";
      in {
        defaultPackage = naersk-lib.buildPackage ./.;
        devShell = with pkgs;
          mkShell {
            buildInputs =
              [
                cargo
                ffmpeg
                iconv
                pre-commit
                python3
                python3Packages.matplotlib
                rustc
                rustfmt
                rustPackages.clippy
              ]
              ++ lib.optionals isMacOS [
                darwin.apple_sdk.frameworks.IOKit
                darwin.apple_sdk.frameworks.QuartzCore
              ]
              ++ lib.optionals isLinux [
                vulkan-loader
                vulkan-headers
                vulkan-validation-layers
                vulkan-tools
                glslang
                shaderc
              ];
            RUST_SRC_PATH = rustPlatform.rustLibSrc;
          };
      }
    );
}
