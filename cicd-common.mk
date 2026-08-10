# Shared, reusable Makefile fragment for cicd_runner-compatible
# pipelines -- self-documenting help target using the well-known
# `## comment` convention: any target line ending in `## text` is
# picked up automatically, no per-project help block to maintain or
# let drift out of sync with the real target list.
#
# Vendored from cicd_runner's examples/cicd-common.mk so that `make
# help` works in a plain checkout of this repo too, not only inside a
# cicd-runner worker (where /etc/cicd-common.mk is baked into the
# image).
#
# Includers must pick ONE of the two paths, not include both. At the
# repo root a worker sees this file (on the bind mount) AND
# /etc/cicd-common.mk at the same time, and including both makes make
# warn about a duplicate `help` recipe -- confirmed live 2026-08-10.
# See the root Makefile's own ifeq guard. core/ is unaffected: a
# worker mounts that directory alone, so its sibling ../ path can
# never resolve there.
help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'
