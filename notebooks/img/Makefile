TEX_FILES=$(wildcard *tex)
PDF_FILES=$(TEX_FILES:%.tex=%.pdf)
SVG_FILES=$(TEX_FILES:%.tex=%.svg)

PDFCMD?=latexmk -xelatex -synctex=1 -interaction=nonstopmode

all: $(PDF_FILES) $(SVG_FILES)

#TODO: using the svg export with the tikz option for standalone doesn't work right
#      need to install texlive-dvisvgm in order to try: latex <filestem> && dvisvgm <filestem>

%.dvi : %.tex
	@latex $<

%.svg : %.pdf
	@inkscape --export-filename=$@ $<

%.pdf : %.tex
	@$(PDFCMD) $<

%.png : %.svg
	@inkscape --export-filename=$@ $<

%.png : %.pdf
	@convert -verbose -density 150 -quality 100 -flatten -trim $< $@

%.jpg : %.pdf
	@convert -verbose -density 150 -quality 100 -flatten -trim $< $@

clean:
	@latexmk -C
	@rm -fv *eps *svg *aux *dvi *log ./*~
	@rm -rv auto
