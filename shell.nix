{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  packages = with pkgs; [
    (python312.withPackages (ps: with ps; [
      numpy
      pandas
      matplotlib
    ]))
  ];

  shellHook = ''
    export PYTHONPATH=$PYTHONPATH
    echo "Bayesian Causal Inference on Crop Yeilds env activated: Python 3.12."
    echo "Run 'python' for scripts, add more packages as needed."
  '';
}
