{
  pkgs,
  lib,
  config,
  inputs,
  ...
}: let
  pkgs-unstable = import inputs.nixpkgs-unstable {system = pkgs.stdenv.system;};
in {
  packages = [
    pkgs.git
    pkgs.tombi
    pkgs.ruff
    pkgs.alejandra
  ];
  cachix.enable = false;
  languages = {
    c = {
      enable = false;
    };
    nix = {
      enable = true;
    };
    python = {
      enable = true;
      version = "3.12";
      uv = {
        enable = true;
        package = pkgs-unstable.uv;
      };
    };
  };

  scripts = let
    uv_run = "${pkgs-unstable.uv}/bin/uv run --active";
    alej_run = "${pkgs.alejandra}/bin/alejandra";
    tombi_run = "${pkgs.tombi}/bin/tombi";
  in {
    run_tests = {
      exec = ''
        devenv tasks run check:local
      '';
    };
    fix_linting = {
      exec = ''
        ${uv_run} ruff format
        ${uv_run} ruff check --fix
        ${alej_run} --exclude ./.devenv --exclude ./.devenv.flake.nix .
        ${tombi_run} format
      '';
    };
  };

  tasks = let
    uv_run = "${pkgs-unstable.uv}/bin/uv run --active";
    uv_build = "${pkgs-unstable.uv}/bin/uv build";
  in {
    "docs:check" = {
      exec = ''
        export LC_ALL=C  # necessary to run in github action
        ${uv_run} sphinx-build -b singlehtml docs build/docs
      '';
      after = ["docs:clean"];
    };
    "docs:build" = {
      exec = ''
        export LC_ALL=C  # necessary to run in github action
        ${uv_run} sphinx-build -j auto -b html docs build/docs
        touch build/docs/.nojekyll  # prevent github from trying to build the docs
      '';
      after = ["docs:clean"];
    };
    "docs:clean" = {
      exec = ''
        rm -rf build/docs/*
      '';
      before = ["docs:build"];
    };
    "test:fast" = {
      exec = ''
        ${uv_run} pytest -m 'not (slow or hardware)'
      '';
    };
    "test:slow" = {
      exec = ''
        ${uv_run} pytest -m 'slow' --reruns 3
      '';
    };
    "test:hardware" = {
      exec = ''
        ${uv_run} pytest -m 'hardware' --reruns 3
      '';
    };
    "test:all" = {
      exec = ''
        ${uv_run} pytest --reruns 3
      '';
    };
    "test:coverage" = {
      exec = ''
        ${uv_run} coverage run -m pytest -m "not hardware" --reruns 3
      '';
    };
    "check:coverage-report" = {
      exec = ''
        ${uv_run} coverage report -m
        ${uv_run} coverage xml
        ${uv_run} coverage html
      '';
      after = ["test:coverage"];
    };
    "check:toml-lint" = {
      exec = ''
        ${uv_run} tombi check .
      '';
    };
    "check:python-lint" = {
      exec = ''
        ${uv_run} ruff check .
      '';
    };
    "check:python-types" = {
      exec = ''
        ${uv_run} ty check .
      '';
    };
    "check:dependencies" = {
      exec = ''
        ${uv_run} pip-audit
      '';
    };
    "check:local" = {
      after = [
        "test:fast"
        "test:slow"
        # "test:hardware"
        "check:python-lint"
        # "check:python-types"
        "check:toml-lint"
      ];
    };
  };
}
