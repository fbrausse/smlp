#!/usr/bin/env bash
gh workflow run Build --repo SMLP-Systems/smlp --ref $(git branch --show-current)
