# LaTeX author support for the NESS society journal *The New England Journal of Statistics in Data Science* (*NEJSDS*)

## Table of Contents

* [About](#about)
* [Package content](#package-content)
* [Setup](#setup)
* [Recomended usage of `nejsds` package](#recomended-usage-of-nejsds-package)
* [Submission](#submission)
* [Bug reports](#bug-reports)

## About

Author support service provides LaTeX style file and `*.tex` file template designed for 
NESS society journal
[The New England Journal of Statistics in Data Science (NEJSDS)](https://journal.nestat.org/sections) articles.

## Package content

The following files are given in the repository (or directly in `*.zip` archive):

* `nejsds.cls` - LaTeX style file.
  Please do not change it. This file is already loaded in the respective template files;
* `nessart-number.bst` - suggested BibTeX[^1] related bibliography style.
  If your bibliography is structured in BibTeX format, loading your `*.bib` file
  and one of provided BibTeX styles allows you to get the final format of the bibliography.
* `nejsds_template.tex` - topmatter template (should be used for article preparation);
* `nejsds_sample.tex` - journal sample article;
* `nejsds_sample.pdf` - journal sample article (`PDF` file);

[^1]: The following link provides information about BibTeX usage: [http://www.bibtex.org/Using/](http://www.bibtex.org/Using/).

## Setup
* Clone the repository or download the `*.zip` archive. Rename the package to `<your-project-name>`.
* Install `nejsds.cls`, `nessart-number.bst` in your TeX system (suggested directory: `nejsds`).
* Use the file `nejsds_template.tex` to start your article as a template.
* Use the file `nejsds_sample.tex` as a reference for how to prepare a topmatter of your article.

## Recommended usage of `nejsds` package

Use `nejsds_template.tex` as a template.

### Dependencies

`nejsds.cls` file loads `flushend.sty` and `stfloats.sty` from [`sttools`](https://ctan.org/pkg/sttools) package.
Please make sure that your TeX system has the latest version of this package.

### Document class options

If you see that some space in the lastpage balanced column is missing you can try
to add \atColsBreak{\pagediscards} which should restore discarded material in original break poin.
Or you can remove flushend with option `noflushend` and balance last column manualy.
For bibliography references output and citations a `natbib` package
is loaded by default with the following options:
```latex
\usepackage[numbers,square]{natbib}
```
It provides numbered citations.

In case author-year citation is required, provide the `authoryear` option:
```latex
\documentclass[authoryear]{nejsds}
```
All `natbib` package options can be provided in this way.

In case some other bibliography package is used
which is not compatible with `natbib` package,
one can disable the latter with the option `nonatbib`:
```latex
\documentclass[nonatbib]{nejsds}
```

### LaTeX document preamble content

The preamble of your LaTeX document should look like this:

```latex
\documentclass{nejsds}

\arxiv{math.PR/0000000}

\begin{document}

    \begin{frontmatter}

        \title{Title\protect\thanksref{T1}}
        \thankstext[id=T1]{Footnote to the title with the `thankstext' command.}

        \begin{aug}
            \author{\inits{F.}\fnms{First} \snm{Author}\thanksref{c1}\ead[label=e1]{first@somewhere.com}}
            \thankstext[type=corresp,id=c1]{Corresponding author.}
            \address{Address of the First Author\\
                     Country\\
                     \printead{e1}}
            \author{\inits{S.}\fnms{Second} \snm{Author}\thanksref{t2}\ead[label=e2]{second@somewhere.com}}
            \address{Address of the Second Author\\
                     Country\\
                     \printead{e2}}
            \author{\inits{T.~N.}\fnms{Third Name} \snm{Author}
                    \ead[label=e3]{third@somewhere.com}%
                    \ead[label=u1,url]{http://www.foo.com}}
            \address{Address of the Third Author\\
                     Country\\
                     \printead{e3}\\
                     \printead{u1}}
            \thankstext[id=t2]{Footnote to the first author with the `thankstext' command.}
        \end{aug}

        \begin{abstract}
            ...
        \end{abstract}

        \begin{keyword}[class=AMS]  % please indicate appropriate AMS codes
            \kwd[Primary ]{00K00}
            \kwd{00K01}
            \kwd[; secondary ]{00K02}
        \end{keyword}

        \begin{keyword}
            \kwd{Sample}
            \kwd{\LaTeX}
        \end{keyword}

        \tableofcontents

    \end{frontmatter}

    Your publication content

\end{document}
```

### Comments

* Labels **T1**, **t2** are used for thanks;
* thanks text `type=corresp` marks corresponding author;
* Labels **e1**, **e2**, **e3**, **u1** are used to print electronic addresses.
`hyperref` package is loaded in the class so they will be made into hyperlinks;

## Submission

Submit one single file as a `ZIP` archive.
Pack your root folder `<your-project-name>` with files and subfolders.

## Bug reports

Please submit bug report or feature requests at
[github](https://github.com/vtex-soft/texsupport.ness-nejsds/issues) page.