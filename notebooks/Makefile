JUPYTEXT_FILES=$(wildcard *py)
NOTEBOOKS_PLAIN=$(JUPYTEXT_FILES:%.py=%.ipynb)
NOTEBOOKS_HTML =$(JUPYTEXT_FILES:%.py=%.html)

PDFCMD?=latexmk -xelatex -synctex=1 -interaction=nonstopmode
JPYOPTS?=--execute

all: $(NOTEBOOKS_PLAIN) $(NOTEBOOKS_HTML)

#TODO: using the svg export with the tikz option for standalone doesn't work right
#      need to install texlive-dvisvgm in order to try: latex <filestem> && dvisvgm <filestem>

%.ipynb : %.py
	@jupytext --to ipynb ${JPYOPTS} $<

%.html : %.ipynb
	@jupyter nbconvert $< --to html

clean:
	@rm -fv $(NOTEBOOKS_PLAIN) $(NOTEBOOKS_HTML)
