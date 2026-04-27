# Build pipeline for paper 1.
#
# Workflow:
#   make sections — convert all ../sections/*.md to sections/*.tex via pandoc
#   make pdf      — runs `make sections` then latexmk -> main.pdf
#   make clean    — removes intermediate latex files
#   make distclean — removes everything generated, including sections/*.tex
#   make arxiv    — produces an arxiv-ready tarball at build/arxiv.tar.gz
#
# Run from the build/ directory:
#   cd papers/accuracy_vs_fragility/build && make pdf

PANDOC      ?= pandoc
LATEXMK     ?= latexmk
TAR         ?= tar

SECTION_MD  := $(wildcard ../sections/*.md)
SECTION_TEX := $(patsubst ../sections/%.md,sections/%.tex,$(SECTION_MD))

MAIN        := main
MAIN_TEX    := $(MAIN).tex
MAIN_PDF    := $(MAIN).pdf

PANDOC_FLAGS := \
  --from=markdown+pipe_tables+grid_tables+raw_tex \
  --to=latex \
  --top-level-division=section \
  --wrap=preserve \
  --template=pandoc-template.tex

# -halt-on-error dropped: latexmk needs multiple passes for forward
# references and bibliography resolution, and `halt` mode trips
# unnecessarily during normal multi-pass builds.
# -f (force) keeps going after pdflatex returns non-zero due to
# unresolved references, which are normal in early passes.
LATEXMK_FLAGS := -pdf -bibtex -f -interaction=nonstopmode

.PHONY: all pdf sections clean distclean arxiv help

all: pdf

# --- Per-section conversion --------------------------------------------
# Pandoc converts markdown -> tex; sed strips the manual heading
# numbering ("# 1. Introduction" -> "# Introduction", "## 4.1 Foo" -> "## Foo",
# "# Appendix A. Foo" -> "# Foo") so LaTeX's \section / \subsection /
# \appendix can auto-number.  Markdown source keeps the manual numbers
# for human readability on github.
sections/%.tex: ../sections/%.md pandoc-template.tex postprocess.py
	@mkdir -p sections
	$(PANDOC) $(PANDOC_FLAGS) -o $@.raw $<
	@sed -E \
	  -e 's/\\section\{[0-9]+\. */\\section{/g' \
	  -e 's/\\subsection\{[0-9]+\.[0-9]+ */\\subsection{/g' \
	  -e 's/\\subsection\{[A-Z]\.[0-9]+ */\\subsection{/g' \
	  -e 's/\\section\{Appendix [A-Z]\. */\\section{/g' \
	  $@.raw > $@
	@python3 postprocess.py $@
	@rm -f $@.raw

sections: $(SECTION_TEX)

# --- Full PDF build -----------------------------------------------------
pdf: sections $(MAIN_PDF)

$(MAIN_PDF): $(MAIN_TEX) $(SECTION_TEX) references.bib
	-$(LATEXMK) $(LATEXMK_FLAGS) $(MAIN)
	@if [ -f $(MAIN_PDF) ]; then \
	  size=$$(wc -c < $(MAIN_PDF)); \
	  pages=$$(grep -c "^Doc-Start" $(MAIN).log 2>/dev/null || echo "?"); \
	  echo "==> $(MAIN_PDF) built ($$size bytes)"; \
	else \
	  echo "ERROR: $(MAIN_PDF) not produced — see main.log"; exit 1; \
	fi

# --- Cleanup ------------------------------------------------------------
clean:
	$(LATEXMK) -c
	rm -f $(MAIN).bbl $(MAIN).blg

distclean: clean
	$(LATEXMK) -C
	rm -rf sections/*.tex
	rm -f arxiv.tar.gz

# --- arXiv tarball ------------------------------------------------------
# Bundle main.tex + bib + sections + figures into a tarball arXiv accepts.
arxiv: pdf
	@mkdir -p arxiv-staging
	cp $(MAIN_TEX) references.bib neurips_2025.sty arxiv-staging/
	cp -r sections arxiv-staging/
	cp -r ../figures arxiv-staging/figures
	$(TAR) -czf arxiv.tar.gz -C arxiv-staging .
	rm -rf arxiv-staging
	@echo "==> arxiv.tar.gz ready for upload"

# --- Help ---------------------------------------------------------------
help:
	@echo "Targets:"
	@echo "  pdf       — build main.pdf (default)"
	@echo "  sections  — convert markdown sections to .tex only"
	@echo "  clean     — remove intermediate latex files"
	@echo "  distclean — remove everything generated"
	@echo "  arxiv     — produce arxiv-ready tarball at arxiv.tar.gz"
