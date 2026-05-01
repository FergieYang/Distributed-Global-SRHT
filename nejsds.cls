\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{nejsds}[2023/12/05 v1.16, driver class for NEJSDS journal]
%% mandatory command for tex4ht proper file load 
\def\fmt@name{nejsds}

\RequirePackage{etoolbox}
%%
%%  >>>>>>> OPTIONS >>>>>>>
%%
%%
%%  ======= GENERAL =======
%%
%%  Load natbib with correct settings
\newif\if@load@natbib@\@load@natbib@true
\DeclareOption{nonatbib}{\@load@natbib@false}

\newif\if@load@flushend@\@load@flushend@true
\DeclareOption{noflushend}{\@load@flushend@false}

\newif\if@numbered@bibl\@numbered@bibltrue
\DeclareOption{authoryear}{\@numbered@biblfalse}

%%  When hyperref is loaded, makes a hyperlink
%%  only from year component of the cite command
\newif\if@linksfromyear\@linksfromyearfalse
\DeclareOption{linksfromyear}{\@linksfromyeartrue}

%%  Put "." after inline section headings
\newif\if@autosecdot
\DeclareOption{autosecdot}{\@autosecdottrue}
\DeclareOption{noautosecdot}{\AtBeginDocument{\@autosecdotfalse}}

\DeclareOption{seceqn}{\AtBeginDocument{\numberwithin{equation}{section}}}

%%  Spacing
\def\singlespacing{\renewcommand{\baselinestretch}{}\large\normalsize}
\def\doublespacing{\renewcommand{\baselinestretch}{1.6}\large\normalsize}
\DeclareOption{doublespacing}{\doublespacing}
\DeclareOption{singlespacing}{\singlespacing}
%%
%%  ======= JOURNAL =======
%%
\PassOptionsToClass{10pt,twocolumn,twoside}{article}


\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions*
\LoadClass{article}

\def\@JID{NEJSDS}
\def\@jid{nejsds}
\def\@journal{The New England Journal of Statistics in Data Science}
\let\@JOURNAL\@journal
\def\@issn{2693-7166}
\def\ip@journal@url#1{https://journal.nestat.org/}
\def\@publisher{New England Statistical Society}
\def\@publisheraddress{}
\def\journal@url{\ip@journal@url{nejsds}}

%%
%%  <<<<<<< OPTIONS <<<<<<<
%%
%%
%%  ======= MAIN SETUP =======
%%

%%
%%  ======= VTeX SETUP =======
%%

\def\set@natbib{%
    \if@linksfromyear
        \AtBeginDocument{\citefix}
    \fi
    \let\bibfont\thebibliography@size
    \setlength\bibsep{\z@}%
    }

\def\load@natbib{%
    \if@numbered@bibl
        \PassOptionsToPackage{numbers,square}{natbib}%
    \fi
    \RequirePackage{natbib}%
    \AtBeginDocument{\set@natbib}%
    }

\RequirePackage{rotating}
    
\IfFileExists{vtexgraphicx.sty}{
      \InputIfFileExists{nejsds-vtex.sty}{}{}
  }{}
  
  
\if@load@natbib@
    \@ifundefined{load@natbib}{}{\load@natbib}%
\fi
  
\RequirePackage{keyval}
\RequirePackage{afterpackage}

\def\do@option@list#1#2{%
    \@for\curr@option:={#2}\do{%
        \expandafter\sep@key@value\curr@option/?/{#1}\relax
        }%
    }
\def\sep@key@value#1=#2/?/#3{\expandafter\gdef\csname #3@#1\endcsname{#2}}

%%  Macro by S.T. for unique enumeration system
\def\ip@setvaluelist#1#2{%
    \@tempcnta=0\relax
    \@for\@curr@val:=#2\do{%
        \advance\@tempcnta by1\relax
        \expandafter\protected@xdef\csname #1@item@\the\@tempcnta\endcsname{\@curr@val}%
        }%
    \expandafter\protected@xdef\csname #1@item@0\endcsname{\the\@tempcnta}%
    }
\xdef\ip@getitemvalue#1#2{\noexpand\csname #1@item@#2\endcsname}

%%  Normal text (justify)
\def\ip@normaltext{%
    \let\\=\@normalcr
    \leftskip\z@
    \@rightskip\z@
    \rightskip\@rightskip
    \parfillskip\@flushglue
    }

\def\no@harm{%
    \let\thanks=\@gobble
    \let\thanksref=\@gobble
    \let~\space
    \def\ead##1##{\@gobble}%
    \let\\=\@empty
    \def\protect{\noexpand\protect\noexpand}%
    }

%%  Hyphenation turn-off
\def\nohyphen{%
    \pretolerance=\@M
    \tolerance=\@M
    \hyphenpenalty=\@M
    \exhyphenpenalty=\@M
    }

%%  \raggedrightmargin{00mm}
\def\raggedrightmargin#1{%
    \let\\\@centercr
    \@rightskip #1 plus 1fil%
    \rightskip\@rightskip
    \leftskip\z@skip
    \parindent\z@\nohyphen
    }

\def\smart@par{%
    \ifhmode
        \par
    \fi
    }

\def\@ifnonempty#1{%
    \setbox\@tempboxa\hbox{\ignorespaces #1}%
    \ifdim\wd\@tempboxa>1pt%
        #1%
    \fi
    }

%%  {prefix}{font}{text}{url}
\def\url@fmt#1#2#3#4{%
    \edef\@tempa{#3}%
    \ifx\@tempa\@empty
    \else
        #1{#2\ip@href{#4}{#3}}%
    \fi
    }
%%  {url}{text}
\def\ip@href#1#2{#2}

\@ifundefined{href}{\let\href\@secondoftwo}{}
%%
%%  ======= PAGE LAYOUT =======
%%
%%  \set@page@layout{\textwidth}{\textheight}
\def\set@page@layout#1#2{%
    \setlength\textwidth{#1}%
    \@settopoint\textwidth
    \setlength\textheight{#2}%
    \@settopoint\textheight
    %%  Make side margins equal
    % \setlength\@tempdima{\paperwidth}%
    % \addtolength\@tempdima{-\textwidth}%
    % \setlength\oddsidemargin{.5\@tempdima}%
    % \addtolength\oddsidemargin{-1in}%
    % \setlength\evensidemargin{\oddsidemargin}%
    % \@settopoint\oddsidemargin
    % \@settopoint\evensidemargin
    %%  Topmargin
    \setlength\topmargin{\paperheight}%
    \addtolength\topmargin{-2in}%
    \addtolength\topmargin{-\headheight}%
    \addtolength\topmargin{-\headsep}%
    \addtolength\topmargin{-\textheight}%
    \addtolength\topmargin{-\footskip}%
    \addtolength\topmargin{-.5\topmargin}%
    \@settopoint\topmargin
    }

\newdimen\t@xtheight
\def\init@settings{%
    \splittopskip=\topskip
    \splitmaxdepth=\maxdepth
    \t@xtheight\textheight
    \advance\t@xtheight-\splittopskip
    }

\setlength\parindent{12\p@}
\setlength\headsep{14\p@}
\setlength\footskip{24\p@}

\setlength\smallskipamount{6\p@ \@plus 1\p@ \@minus 1\p@}
\setlength\medskipamount{12\p@ \@plus 3\p@ \@minus 3\p@}
\setlength\bigskipamount{18\p@ \@plus 3\p@ \@minus 3\p@}

\set@page@layout{44pc}{55pc}
%%  Side margins
\setlength\oddsidemargin{0.7in} %  gutter margin
\oddsidemargin=\dimexpr 0.7in-1in\relax %  outer margin
\@tempdima=\paperwidth
\advance\@tempdima by-\oddsidemargin
\advance\@tempdima by-\textwidth
\evensidemargin=\dimexpr\@tempdima-2in\relax %  outer margin

%%
%%  ======= FONT =======
%%
%%  Change to main font
\renewcommand\normalsize{%
    \@setfontsize\normalsize\@xpt{12\p@ plus .3\p@ minus .3\p@}%
    \abovedisplayskip 10\p@ \@plus2\p@ \@minus2\p@
    \abovedisplayshortskip 6\p@ \@plus2\p@
    \belowdisplayshortskip 6\p@ \@plus2\p@
    \belowdisplayskip \abovedisplayskip
    \let\@listi\@listI
    }

\PassOptionsToPackage{cmex10}{amsmath}
\RequirePackage{amsmath}
\allowdisplaybreaks
\@addtoreset{equation}{section}
\renewcommand\theequation{\thesection.\@arabic\c@equation}

\RequirePackage{amsthm}
\RequirePackage{amssymb}
\RequirePackage{amsfonts}
%%
%%  ======= HEADERS =======
%%
%%  \pagestyle{headings} \thispagestyle{copyright}
\def\runninghead@size{\small\itshape}
\def\pagenumber@size{\small\upshape}

\def\ps@headings{%
    \def\pseudo@head@line{%
        \runninghead@size\rightmark
        \hbox to18\p@{\hfill\pagenumber@size 000}%
        }%
    \def\@oddhead{%
        \csname check@head@line\endcsname
        \runninghead@size
        \hfill\rightmark
        \hbox to18\p@{\hfill\pagenumber@size\thepage}%
        }%
    \def\@evenhead{%
        \runninghead@size
        \hbox to18\p@{\pagenumber@size\thepage\hfill}%
        \leftmark\hfill
        }%
    \def\@evenfoot{}%
    \def\@oddfoot{}%
    }

\def\check@head@line{%
    \bgroup
        \settowidth\@tempdima{\pseudo@head@line}%
        \ifdim\@tempdima>\textwidth\relax
            \ClassError{nejsds}{Running title is too long!}{}%
            \global\let\check@head@line\relax
        \fi
    \egroup
    }

\def\copyright@size{\sffamily\small\raggedright}

\def\ps@copyright{%
    \let\@mkboth\@gobbletwo
    \def\@evenhead{%
        \smash{%
            \lower24\p@ \vbox{\hsize=\textwidth \copyright@size\copyright@text}%
            }%
        }%
    \let\@oddhead\@evenhead
    \def\@oddfoot{\hfill\pagenumber@size\thepage\hfill}%
    \let\@evenfoot\@oddfoot
    }

\def\copyright@text{%
    \url@fmt{}{\copyright@url@font}{\@journal}{\journal@url}%
    \enskip
    \@ifnonempty{\@volume, \@pagerange\ (\@pubyear)}%
    \print@copyrightowner\break
    \print@doi
    }
\def\copyright@url@font{\scshape}
\def\doi@text{\url@fmt{url: }{\ttfamily}{\paper@url}{\paper@url}}
\def\paper@url{}

%%  copyrightowner
\def\print@copyrightowner{%
    \hfill
    \if!\copyrightowner@text!%
    \else
        {\textcopyright\space\scshape\copyrightowner@text}%
    \fi
    }
\def\copyrightowner#1{\def\copyrightowner@text{#1}}
\def\copyrightowner@text{}

\def\copyright@fmt{%
    \@ifundefined{\copyrightowner@text}{}{\safe@footnotetext{\copyright@owner}}%
    }
\def\copyright@owner{$\copyright$~\@copyrightyear \copyrightowner@text}

\def\add@tok#1#2{%
  \global#1\expandafter{\the#1#2}%
  }
\def\add@xtok#1#2{%
  \begingroup
    \no@harm
    \xdef\@act{%
      \global\noexpand#1{\the#1#2}%
      }%
    \@act
  \endgroup
  }

%%
%%  ======= PREAMBLE =======
%%
  
\def\doiprefix#1{\gdef\doi@prefix{#1}}
\def\doi@prefix{DOI:~}
\@ifundefined{doi@base}{\def\doi@base{https://doi.org/}}{}
\def\print@doi{\doi@prefix\href{\doi@base\@doi}{\doi@base\@doi}}

\csdef{@volume}{0}
\csdef{@pubyear}{0000}
\csdef{@@ppages}{\empty}
\csdef{@firstpage}{0}
\csdef{@lastpage}{0}
\csdef{@doi}{\empty}
\def\doc@stage{300}

\def\volume#1{\gdef\@volume{#1}}
\def\issue#1{\gdef\@issue{#1}}
\def\pubyear#1{\gdef\@pubyear{#1}}
\def\doi#1{\gdef\@doi{#1}}

\def\@vtexed{??}
\def\vtexed#1{\def\@vtexed{#1}}

\def\@articletype{??}
\def\articletype#1{\gdef\@articletype{#1}}
\@onlypreamble\articletype
\def\aid#1{\gdef\@aid{#1}}

%%  pubyear, volume, paperno
\def\@pubyear{}
\def\@copyrightyear{}
\def\pubyear#1{%
    \gdef\@pubyear{#1}%
    \gdef\@copyrightyear{#1}%
    }

\def\@volume{}
\def\volume#1{\gdef\@volume{Volume #1}}

\def\@issue{}
\def\issue#1{\gdef\@issue{Number #1}}

%%  arXiv
\def\arxiv#1{%
    \gdef\@arxiv{#1}%
    \gdef\doi@text{%
        \url@fmt{arXiv: }{\ttfamily}{#1}{\arxiv@base\@arxiv}%
        }%
    }
\let\@arxiv\relax
%%  http://arxiv.org/abs/math.PR/0603300
\def\arxiv@base{http://arxiv.org/abs/}

%%  firstpage, lastpage, pagerange

\IfFileExists{nejsds-vtex.sty}{}{
  \RequirePackage{lastpage}
  }

\def\set@default@pages{%
    \ifnum\@firstpage=0\relax
      \firstpage{1}%
    \fi
    \@ifundefined{lastpage@lastpage}{}{%
      \def\@temp@relax{\relax}%
      \ifx\lastpage@lastpage\@temp@relax
      \else
        \ifnum\@lastpage=0\relax
          \expandafter\lastpage\expandafter{\lastpage@lastpage}%
        \fi
      \fi
    }%
}

    
\AtBeginDocument{%
  \set@default@pages
  }
    
\newcounter{firstpage}
\newcounter{lastpage}


\def\firstpage#1{%
    \def\@tempa{#1}%
    \ifx\@tempa\@empty
    \else
        \global\c@firstpage=#1%
        \global\c@lastpage=#1%
        \global\c@page=#1%
        \gdef\@firstpage{#1}%
        \ignorespaces
    \fi
    }
\def\lastpage#1{%
    \def\@tempa{#1}%
    \ifx\@tempa\@empty
    \else
        \global\c@lastpage=#1%
        \gdef\@lastpage{#1}%
        \ignorespaces
    \fi
    }

\def\@pagerange{}
\def\set@pagerange{%
    \ifnum\c@firstpage=0%
    \else
        \ifnum\c@firstpage=\c@lastpage
            \gdef\@pagerange{\thefirstpage}%
        \else
            \gdef\@pagerange{%
                \thefirstpage\pagerange@sep\thelastpage
                }%
        \fi
    \fi
    }
\def\pagerange@sep{--}

\def\pagenumbering#1{%
    \gdef\thefirstpage{\csname @#1\endcsname\c@firstpage}%
    \gdef\thelastpage{\csname @#1\endcsname\c@lastpage}%
    \gdef\thepage{\csname @#1\endcsname\c@page}%
    }
%%  hyperref redefines \pagenumbering, so we must override hyperref definition
\let\ip@pagenumbering\pagenumbering

%%  startlocaldefs, endlocaldefs
\def\startlocaldefs{\makeatletter}
\def\endlocaldefs{\makeatother}
%%
%%  ======= HISTORY DATES =======
%%
%%  received, revised, accepted
\def\received@prefix{Received~}
\def\revised@prefix{; revised~}
\def\accepted@prefix{Accepted~}

\def\@publishedonline{}
\def\publishedonline#1{\gdef\@publishedonline{#1}}

\def\history@exist{0}
\def\@received{\@nil}
\def\received#1{%
    \def\@tempa{#1}%
    \ifx\@tempa\@empty
    \else
        \gdef\@received{#1}%
        \gdef\history@exist{1}%
    \fi
    }
\def\@revised{\@nil}
\def\revised#1{%
    \def\@tempa{#1}%
    \ifx\@tempa\@empty
    \else
        \gdef\@revised{#1}%
        \gdef\history@exist{1}%
    \fi
    }
\def\@accepted{\@nil}
\def\accepted#1{%
    \def\@tempa{#1}%
    \ifx\@tempa\@empty
    \else
        \gdef\@accepted{#1}%
        \gdef\history@exist{1}%
    \fi
    }

\def\empty@data{\@nil}

\def\history@style{\centering}
\def\history@font{\normalsize\itshape\mdseries}
\def\history@skip{\medskipamount}
\def\history@postfix{}
\def\history@fmt{%
    \ifcase\history@exist
    \else
        \bgroup
            \nobreak
            \vskip\history@skip
            \nobreak
            \history@style
            \history@font
            \parindent=\z@
            \leavevmode
            \ifx\@received\empty@data
            \else
                \received@prefix\@received
            \fi
            \ifx\@revised\empty@data
            \else
                \revised@prefix\@revised
            \fi
            \ifx\@accepted\empty@data
            \else
                \accepted@prefix\@accepted
            \fi
            \history@postfix
            \par
        \egroup
        \gdef\history@exist{0}%
    \fi
    \global\let\history@fmt\relax
    }
    
    
\def\sday#1{#1}
\def\smonth#1{%
    \@ifundefined{month@item@#1}%
        {\@latex@error{Incorrect month value #1!}{??}}%
        {\ip@getitemvalue{month}{#1}}%
    }%
\def\syear#1{#1}
\ip@setvaluelist{month}{%
    January,February,March,April,May,June,July,%
    August,September,October,November,December%
    }
\AtBeginDocument{%
    \let\sv@thebibliography\thebibliography
    \def\thebibliography{\history@fmt\sv@thebibliography}%
    }
%%
%%  >>>>>>> FRONTMATTER >>>>>>>
%%
\def\author@num{0}
\newcounter{author}
\newcounter{address}
\newdimen\sv@mathsurround
\newdimen\sv@parindent
\sv@parindent\parindent
\newbox\fm@box
\newdimen\fm@size

\let\hy@frontmatter\relax
\let\hy@endfrontmatter\relax

\def\frontmatter@style{\raggedright}

\def\frontmatter{%
    \global\c@author\z@
    \global\c@address\z@
    \renewcommand\thefootnote{\@fnsymbol\c@footnote}%
    \def\pdftitle##1{\write@pdfinfo{\user@hy@title}{##1}}%
    \def\pdfauthor##1{\write@pdfinfo{\user@hy@author}{##1}}%
    \def\pdfsubject##1{\write@pdfinfo{\user@hy@subject}{##1}}%
    \def\pdfkeywords##1{\write@pdfinfo{\user@hy@keywords}{##1}}%
    \sv@mathsurround\mathsurround \m@th
    \parindent\z@
    \hy@frontmatter
    \global\let\maketitle\relax
    \open@fm
        \ignorespaces
    }
\def\endfrontmatter{%
        \global\@topnum\z@
        \set@pagerange
        \csname form@runauthors\endcsname
        \markboth{\@runauthor}{\@runtitle}%
        \thispagestyle{copyright}%
        \put@fmt@data
    \close@fm
    \firstpage@cmd
    \write@pdfinfo{\hy@author}{\the\authors@list}%
    \write@pdfinfo{\hy@subject}{\@journal, \@copyrightyear, \@volume, \@issue, \@pagerange}%
    \write@pdfinfo{\hy@keywords}{\the\keywords@list}%
    \write\@mainaux{\string\gdef\string\author@num{\the\c@author}}%
    \hy@endfrontmatter
    \global\mathsurround\sv@mathsurround
    \global\c@footnote\z@
    \global\let\@thanks\@empty
    \let\title\relax
    \let\author\relax
    \let\address\relax
    \let\frontmatter\relax
    \let\endfrontmatter\relax
    \let\@maketitle\relax
    \let\@@maketitle\relax
    \aftergroup\frontmatter@cmd
    }

\def\open@fm{%
    \global\setbox\fm@box=\vbox\bgroup
        \hsize=\textwidth
        \frontmatter@style
    }
\def\close@fm{%
        \par
        \ifvoid\abstract@box
        \else
            \hrule height1\p@
            \@tempswatrue
        \fi
        \abstract@fmt
        \keyword@fmt
        \if@tempswa
            \medskip
            \hrule height1\p@
        \fi
        \vskip\baselineskip
    \egroup
    \fm@size=\dp\fm@box
    \advance\fm@size by \ht\fm@box
    \@whiledim\fm@size>\t@xtheight\do{%
        \global\setbox\@tempboxa=\vsplit\fm@box to \t@xtheight
        \unvbox\@tempboxa
        \fm@size=\dp\fm@box
        \advance\fm@size by \ht\fm@box
        }%
    \if@twocolumn
        \emergencystretch=1pc%
        \twocolumn[\unvbox\fm@box]%
    \else
        \unvbox\fm@box
    \fi
    }

%%  \maketitle
%%  If \frontmatter is not used, we will redefine \maketitle
\def\local@maketitle{%
    \global\@topnum\z@
    \set@pagerange
    \markboth{\@runauthor}{\@runtitle}%
    \thispagestyle{copyright}%
    \put@fmt@data
    \write@pdfinfo{\hy@author}{\the\authors@list}%
    \write@pdfinfo{\hy@keywords}{\the\keywords@list}%
    \hy@endfrontmatter
    \global\mathsurround\sv@mathsurround
    \global\c@footnote\z@
    \global\let\@thanks\@empty
    \let\title\relax
    \let\author\relax
    \let\address\relax
    \let\frontmatter\relax
    \let\endfrontmatter\relax
    \let\@maketitle\relax
    \let\@@maketitle\relax
    \normalfont\ip@normaltext
    \parindent\sv@parindent
    \frontmatter@cmd
    }
\AtBeginDocument{\let\maketitle\local@maketitle}

\def\frontmatter@cmd{}

\def\firstpage@cmd{%
    \ip@tableofcontents
    \@ifundefined{@arxiv}%
        {}{\safe@footnotetext{\url@fmt{arXiv: }{}{\@arxiv}{\arxiv@base\@arxiv}}}%
    \markboth{\@runauthor}{\@runtitle}%
    \@thanks
    }

\def\put@fmt@data{%
    \vskip18\p@
    }

%% >>> init FM element setup >>>
\newcommand\init@fm@element[3][]{%
  \def\fm@object{#2}%
  \ifcsname init@fm@#2@done\endcsname
  \else
    \setkeys{#2}{#1}%
  \fi
  \csgdef{init@fm@#2@done}{1}%
  \proc@elem{#2}{#3}%
  }
%% <<< init FM element setup <<<
    
\def\pretitle{\@ifnextchar[\@@pretitle{\@@pretitle[]}}
\def\@@pretitle[#1]#2{\gdef\@pretitle{#2}}
    
%%
%%  ======= TITLE =======
%%

\def\runtitle#1{\gdef\@runtitle{#1}}
\def\@runtitle{}

\def\title@style{\centering}
\def\title@font{\LARGE\sffamily\bfseries}
\def\title@skip{36\p@}
    
%% >>> title/subtitle setup >>>
\def\title#1{%
  \beg@elem
  \init@fm@element{title}{#1}%
  \gdef\@runtitle{#1}%
  \protected\csxdef{title@t@loc@notes}{\the\t@loc@notes}%
  \title@fmt{#1}{\csuse{title@t@loc@notes}}%
  \csname title@end@hook\endcsname  
  }
\def\title@fmt#1#2{%
    \bgroup
        \no@harm
        \xdef\@argi{#1}%
        \xdef\@title{#1}%
    \egroup
    \write@pdfinfo{\hy@title}{\@argi}%
    \bgroup
        \vglue\title@skip
        \title@style\title@font#1{\hbox{\title@textsuperscript{#2}}}\par
    \egroup
  }
  
\def\subtitle#1{%
  \beg@elem
  \init@fm@element{subtitle}{#1}%
  \protected\csxdef{subtitle@t@loc@notes}{\the\t@loc@notes}%
  \csname subtitle@end@hook\endcsname
  }
\def\subtitle@fmt#1#2{%
  \bgroup
    \nohyphen\raggedright
    \subtitle@font
    #1\hbox{\title@textsuperscript{#2}}%
  \egroup
  }

\def\title@textsuperscript#1{\textsuperscript{#1}}
    
    
%%
%%  ======= THANKS =======
%%
%%  thanksref, thanksmark, thankslabel, thankstext
%%  To be safe with hyperref we will use original LaTeX definitions
\def\saferef#1{%
    \expandafter\safe@setref\csname r@#1\endcsname\@firstoftwo{#1}%
    }
\let\safe@setref\@setref

\let\thanksnewlabel\newlabel
\def\safelabel#1{%
    \@bsphack
        \protected@write\@auxout{}%
            {\string\thanksnewlabel{#1}{{\@currentlabel}{\thepage}}}%
    \@esphack
    }

\long\def\safe@footnotetext#1{%
    \insert\footins{%
        \reset@font\footnotesize
        \interlinepenalty\interfootnotelinepenalty
        \splittopskip\footnotesep
        \splitmaxdepth \dp\strutbox
        \floatingpenalty \@MM
        \hsize\columnwidth
        \@parboxrestore
        \color@begingroup
            \def\@thefnmark{}%
            \@makefntext{%
                \rule\z@
                \footnotesep\ignorespaces#1%
                \@finalstrut\strutbox
                }%
        \color@endgroup
        }%
    }

\long\def\orig@footnotetext#1{%
    \insert\footins{%
        \reset@font\footnotesize
        \interlinepenalty\interfootnotelinepenalty
        \splittopskip\footnotesep
        \splitmaxdepth \dp\strutbox
        \floatingpenalty \@MM
        \hsize\columnwidth
        \@parboxrestore
        \protected@edef\@currentlabel{%
            \csname p@footnote\endcsname\@thefnmark
            }%
        \color@begingroup
            \@makefntext{%
                \rule\z@
                \footnotesep\ignorespaces#1%
                \@finalstrut\strutbox
                }%
        \color@endgroup
        }%
    }


%% >>> \beg@elem, \proc@elem (based on iosart2x.cls) >>>
\def\beg@elem{%
  \global\t@loc@notes={}%
  \global\note@cnt\z@
  }
\def\@xnamedef#1{%
  \expandafter\xdef\csname #1\endcsname
  }
\def\no@harm{%
  \let\\=\relax  \let\rm\relax
  \let\ss=\relax \let\ae=\relax \let\oe=\relax
  \let\AE=\relax \let\OE=\relax
  \let\o=\relax  \let\O=\relax
  \let\i=\relax  \let\j=\relax
  \let\aa=\relax \let\AA=\relax
  \let\l=\relax  \let\L=\relax
  \let\d=\relax  \let\b=\relax \let\c=\relax
  \let\bar=\relax
  \def\qq@group@start{}\def\qq@group@end{}\let\qq@cmd\@firstofone\def\qq@style{}%
  \def\protect{\noexpand\protect\noexpand}%
  }
\def\thanks#1#{\@gobble}%
\def\thanksref#1#{\@gobble}%

\def\proc@elem#1#2{%
  \begingroup
    \no@harm
    \def\thanks##1##{\@gobble}%
    \def\thanksref##1##{\@gobble}%
    \csgdef{@#1}{#2}%
  \endgroup
  \prev@elem=\cur@elem
  \cur@elem=\csname e@#1\endcsname
  %\expandafter
  \elem@nothanks#2\thanks\relax
  %\expandafter
  \elem@nothanksref#2\thanksref\relax
  }
%% <<< \beg@elem, \proc@elem (based on iosart2x.cls) <<<

%% >>> thanks macro block >>>
\newtoks\t@glob@notes
\newtoks\t@loc@notes
\newcount\note@cnt
\newcount\c@ThanksID
\newcount\prev@elem \prev@elem=0
\newcount\cur@elem  \cur@elem=0
\chardef\e@pretitle=1
\chardef\e@title=1
\chardef\e@subtitle=1
\chardef\e@author=2
\chardef\e@address=3

\newif\if@thankshyperlink \global\@thankshyperlinkfalse

\def\add@tok#1#2{%
  \global#1\expandafter{\the#1#2}%
  }
\def\add@xtok#1#2{%
  \begingroup
    \no@harm
    \xdef\@act{%
      \global\noexpand#1{\the#1#2}%
      }%
    \@act
  \endgroup
  }
    
\newcounter{correspnote}
\def\corresp@note@counter{correspnote}
\def\corresp@posthook{}
\def\corresp@prehook{}

\newcounter{titlenote}
\def\title@posthook{}
\def\title@prehook{}

\def\title@note@counter{titlenote}
\def\subtitle@note@counter{titlenote}
\def\relatedarticle@note@counter{titlenote}
\def\correctionnote@note@counter{titlenote}
\def\correctedarticle@note@counter{titlenote}
\def\commentaryarticle@note@counter{titlenote}

\newcounter{authornote}
\def\author@note@counter{authornote}
\def\theauthornote{\arabic{authornote}}%
\def\thecorrespnote{%
  \ifcase\c@correspnote\relax
  \or
    $\ast$%
  \or
    $\ast\ast$%
  \or
    $\ast\ast\ast$%
  \fi
  }%
\def\thetitlenote{%
  \ifcase\c@titlenote\relax
  \or
    \protect\ding{73}%
  \or
    \protect\ding{73}\protect\ding{73}%
  \or
    \protect\ding{73}\protect\ding{73}\protect\ding{73}%
  \fi
  }%
\let\thesubtitlenote\thetitlenote

\let\thanks@id\@empty
\let\thanks@mark\@empty
\let\thanks@type\@empty
\define@key{thanks}{id}{\def\thanks@id{#1}}
\define@key{thanks}{mark}{\def\thanks@mark{#1}}
\define@key{thanks}{type}{\def\thanks@type{#1}}
\def\reset@thanks{%
  \let\thanks@id\@empty
  \let\thanks@mark\@empty
  \let\thanks@type\@empty
  }
\def\thanks@optarg@fmt#1#2{%
  \freefootnotetext[#1]{#2}%
  }
\def\thanks@optarg[#1]#2{%
  \let\thanks@type\@empty
  \setkeys{thanks}{#1}%
  \ifx\thanks@type\@empty
    \let\thanks@type\fm@object
  \fi
  \edef\current@counter{\csname\thanks@type @note@counter\endcsname}%
  \refstepcounter{\current@counter}%
  \setcounter{footnote}{\the\csname c@\current@counter\endcsname}%
  \def\thefootnote{\csname the\current@counter\endcsname}%
  \edef\thanks@optarg@id{\thanks@type.thanks\the\c@footnote.\jobname}%
  \@@@Label{\thanks@optarg@id}%
  \if@thankshyperlink %% TODO: make active links
    \ifx\thanks@id\@empty%
      \Hy@GlobalStepCount\Hy@linkcounter
      \xdef\thanks@currentHref{\thanks@type.thanks\the\c@footnote.\jobname.\the\Hy@linkcounter}%
      \edef\my@hy@tmp{\noexpand\hypertarget{\thanks@currentHref}{}}%
    \else
      \edef\my@hy@tmp{\noexpand\hypertarget{\thanks@id thanks}{}}%
    \fi
    \ifx\thanks@mark\@empty
      \def\thanks@mark{\thefootnote}%
    \fi
      \freefootnotetext[\thanks@mark]{{\my@hy@tmp #2}}%
  \else
    \ifx\thanks@mark\@empty
      \def\thanks@mark{\thefootnote}%
    \fi
    \thanks@optarg@fmt{\thanks@mark}{#2}%
  \fi
  \csname end@thanks@optarg@hook\endcsname
  \ignorespaces
  }%
\def\freefootnotetext[#1]{%
  \begingroup
    \unrestored@protected@xdef\@thefnmark{#1}%
  \endgroup
  \@footnotetext
  }
  
\def\elem@nothanksref#1\thanksref{%
  \futurelet\@peektok\elem@thanksref
  }
\def\elem@thanksref{%
  \ifx\@peektok\relax
  \else
    \expandafter\elem@morethanksref
  \fi
  }
\def\elem@morethanksref#1{%
  \csxdef{#1@thankstext@type}{\fm@object}%
  \csletcs{thankstext@type}{#1@thankstext@type}%
  \add@thanks{id=#1}%
  \elem@nothanksref
  }
\def\elem@nothanks#1\thanks{%
  \futurelet\@peektok\elem@thanks
  }
\def\elem@thanks{%
  \ifx\@peektok\relax
  \else
    \ifx\@peektok[%
      \expandafter\expandafter\expandafter\elem@morethankse
    \else
      \expandafter\expandafter\expandafter\elem@morethanks
    \fi
  \fi
  }
\def\elem@morethankse[#1]#2{%
  \thanks@optarg[#1]{#2}%
  \add@thanks{#1}%
  \elem@nothanks
  }
\def\elem@morethanks#1{%
  \thanks@optarg[]{#1}%
  \add@thanks{}%
  \elem@nothanks
  }

\define@key{thankstext}{type}{\gdef\thankstext@type{#1}}
\define@key{thankstext}{id}{\gdef\thankstext@id{#1}}
\def\reset@thankstext{%
  \let\thankstext@id\@empty
  \let\thankstext@mark\@empty
  \let\thankstext@type\@empty
  }
\def\thankstext@fmt#1#2{%
  \freefootnotetext[#1]{#2}%
  }
\def\thankstext[#1]#2{%
    \reset@thankstext
    \setkeys{thankstext}{#1}%
    \ifx\thankstext@type\@empty
      \csletcs{thankstext@type}{\thankstext@id @thankstext@type}%
    \fi
    \edef\current@counter{\csname\thankstext@type @note@counter\endcsname}%
    \refstepcounter{\current@counter}%
    \setcounter{footnote}{\the\csname c@\current@counter\endcsname}%
    \def\thefootnote{\csname the\current@counter\endcsname}%
    \@@@Label{\thankstext@id}%
     \protected@xdef\@thanks{%
        \@thanks
        \protect\thankstext@fmt{\thefootnote}{%
          % \expandonce{\csname \thankstext@type @prehook\endcsname}%
          \protect\csuse{\thankstext@type @prehook}%
          #2%
          % \expandonce{\csname \thankstext@type @posthook\endcsname}}%
          \protect\csuse{\thankstext@type @posthook}}%
      }%
  }
 
\define@key{addthanks}{id}{\def\addthanks@id{#1}}
\define@key{addthanks}{mark}{\def\addthanks@mark{#1}}
\define@key{addthanks}{type}{\def\addthanks@type{#1}}
\def\reset@addthanks{%
  \let\addthanks@id\@empty
  \let\addthanks@mark\@empty
  \let\addthanks@type\@empty
  }
\def\add@thanks#1{%
  \reset@addthanks
  \setkeys{addthanks}{#1}%
  \global\advance\note@cnt\@ne
  \ifnum\note@cnt>\@ne
    \add@xtok\t@loc@notes{\note@sep}%
  \fi
  \if@thankshyperlink %% TODO: make active links
    \ifx\addthanks@id\@empty
      \add@xtok\t@loc@notes{\protect\Hy@breaklinkstrue
        \protect\hyper@linkstart{link}{\thanks@currentHref}\thefootnote\protect\hyper@linkend}%
    \else
      \add@xtok\t@loc@notes{\protect\Hy@breaklinkstrue
        \protect\hyper@linkstart{link}{\addthanks@id thanks}\addthanks@id\protect\hyper@linkend}%
    \fi%
  \else
    \ifx\addthanks@id\@empty
      \def\addthanks@mark{\thefootnote}%
    \else
      \ifx\addthanks@mark\@empty
        \def\addthanks@mark{\@@@Ref{\addthanks@id}}%
      \fi
    \fi
    \add@xtok\t@loc@notes{\addthanks@mark}%
  \fi
  \csname end@add@thanks@hook\endcsname
  }

\def\@@@Label#1{%
  \@bsphack
  \protected@write\@auxout{}{\string\Newlabel{#1}{\@currentlabel}}%
  \@esphack
  }
\def\Newlabel#1#2{%
  \expandafter\gdef\csname X@#1\endcsname{#2}%
  }
\def\@@@Ref#1{%
  \@ifundefined{X@#1}
    {0}
    {\csname X@#1\endcsname}%
  }
  
%%
%%  ======= AUTHORS =======
%%
%%  aug - author block
\def\author@font{\large}
\def\author@preskip{12\p@}
\newif\if@firstauthor
\newif\if@newelem
\let\cmd@for@aut@elem\@gobblethree
\let\mdforvkt\@gobblethree
\let\orcid\@gobble

\newenvironment{aug}{}{%
  \output@frontmatter@authors
  \smart@par}

%%  Auto-generate \runauthor
\newtoks\runauthors@list
\newtoks\one@runauthors@list
\def\runauthor#1{{%
    \def\etal{et al.}%
    \gdef\@runauthor{#1}%
    }}
\def\@runauthor{%
    \ifnum\theauthor<4\relax
        \the\runauthors@list
    \else
        \the\one@runauthors@list\ \@etal
    \fi
    }
\def\@etal{et al.}

\let\bio\@gobble
\def\prefix#1{#1}
\def\degs#1{#1}
\protected\def\inits#1{\xdef\@inits{#1}}
\def\fnms#1{#1}
\def\snm#1{#1}
\def\roles#1{#1}
\def\and{\unskip~and~}
\def\author@sep{}

\def\addr@author@sep{}
\def\author@group{}
\def\author@style{\centering}
\def\author@font{\large\scshape}
\def\authors@skip{16\p@}
    
\def\author{%
  \@ifnextchar[%
    {\author@optarg}
    {\author@optarg[]}%
  }
  
\protected\def\fmt@snm#1{\uppercase{#1}}
\protected\def\fmt@fnms#1{#1}
\let\write@aux@au\@gobbletwo
\protected\def\@write@aux@au#1#2{%
  \protect\protected@write\@auxout{}{\string\global\string\@namedef{#1@\the\c@author @metadata}{#2}}%
  }
\protected\def\inforbio#1{\csxdef{dg@bio@#1@label}{1}}

\def\front@author@defs{%
  \let\bio\inforbio
  \def\ead##1##{\@gobble}%
  \let\write@aux@au\@write@aux@au
  \protected\def\degs##1{##1}%
  \protected\def\prefix##1{%
    \write@aux@au{prefix}{##1}%
    \fmt@prefix{##1}%
    }%
  \protected\def\fnms##1{\fmt@fnms{##1}}%
  \protected\def\inits##1{%
    \csdef{@inits}{##1}%
    \write@aux@au{inits}{##1}%
    }%
  \protected\def\snm##1{%
    \csdef{@snm}{##1}%
    \write@aux@au{snm}{##1}%
    \fmt@snm{##1}%
    }%
  \protected\def\suffix##1{##1}%
  \protected\def\roles##1{##1}%
  }

  
\def\author@optarg[#1]#2{%
  \stepcounter{author}%
  \beg@elem
  %% No need references to address (it is only for xml)
  % \@for\@tempa:=#1\do{\expandafter\add@addressref\expandafter{\@tempa}}%
  \let\oldunderscore\_%
  \let\_\textunderscore
  \init@fm@element{author}{#2}%
  \let\_\oldunderscore
  \temptoks=\expandafter{\@author}%
  \csxdef{author\the\c@author}{\expandonce{\the\temptoks}}%
  \protected\csxdef{author\the\c@author t@loc@notes}{\the\t@loc@notes}%
  \@ifundefined{author@list}
    {%
      \xdef\author@list{author\the\c@author}%
    }{%
      \xappto\author@list{,author\the\c@author}%
    }%
  \csname author@end@hook\endcsname
  }%

\newtoks\pdfmeta@authors
\def\addto@pdfmeta@authors#1{%
  \begingroup
    \xdef\@act{\global\noexpand\pdfmeta@authors{\the\pdfmeta@authors#1}}%
    \@act
  \endgroup
  }
\def\do@pdfmeta@authors{%
  \@tempcnta\z@
  \@whilenum\@tempcnta<\c@author\do{%
    \advance\@tempcnta\@ne
    \ifnum\@tempcnta>\@ne
      \addto@pdfmeta@authors{, }%
    \fi
    \addto@pdfmeta@authors{%
      \@nameuse{inits@\the\@tempcnta @metadata}%
      \@ifundefined{inits@\the\@tempcnta @metadata}
        {}
        { }\@nameuse{snm@\the\@tempcnta @metadata}%
      }%
    }%
  }

\newtoks\temptoks
\def\author@fmt#1#2#3{%
  \ifdefined\author@fmt@init@done
  \else
    \author@fmt@init
  \fi
  \edef\@tempb{#2}%
  \ifx\@tempb\@empty
    #3%
  \else
    %% no need links on author marks
    \let\hyper@@link\@gobbletwo
    #3\fm@textsuperscript{#2}%
  \fi
  }
\def\fm@textsuperscript#1{%
  \unskip\textsuperscript{{\normalfont #1}}%
  }

\def\author@fmt@init{%
  \gdef\author@fmt@init@done{1}%
  \vskip\authors@skip
  \noindent
  \author@style\author@font
  }

\def\author@note@fmt{%
  \def\thefootnote{\sxarabic{footnote}}}
    
%%  Auto-generate \runauthor
\newtoks\authors@list
\def\addto@authors@list#1{%
    \begingroup
        \let\inits\@gobble
        \let\fnms\@firstofone
        \no@harm
        \xdef\@act{%
            \global\noexpand\authors@list{\the\authors@list#1}%
            }%
        \@act
    \endgroup
    }
\def\addto@runauthors@list#1{%
    \begingroup
        \let\inits\@firstofone
        \let\fnms\@gobble
        \def\ead##1##{\@gobble}%
        \no@harm
        \xdef\@runact{%
            \global\noexpand\runauthors@list{\the\runauthors@list#1}%
            }%
        \@runact
        \ifnum\theauthor=1\relax
            \xdef\@runact{%
                \global\noexpand\one@runauthors@list{\the\one@runauthors@list#1}%
                }%
            \@runact
        \fi
    \endgroup
    }

\def\sxarabic#1{%
  \expandafter
  \ifcase\value{#1}%
  \or *%
  \or **%
  \or ***%
  \or ****%
  \or *****%
  \fi
  }

\def\add@addressref#1{%
  \global\advance\note@cnt\@ne
  \ifnum\note@cnt>\@ne
    \add@xtok\t@loc@notes{\note@sep}%
  \fi
  \add@tok\t@loc@notes{\ref{aff.#1}}%
  }
\def\note@sep{,}

\def\output@frontmatter@authors{%
  \edef\authors@num{\the\c@author}%
  \c@author=0
  \def\do##1{%
    \advance \c@author by 1
    \ifnum\c@author>1
      \ifnum\authors@num=2\relax
        \def\author@sep{\unskip\ and }%
      \else  
          \ifnum\c@author=\authors@num\relax
            \def\author@sep{\unskip, and }%
          \else
            \def\author@sep{\unskip,\space}%
          \fi
      \fi
    \fi
    \addto@runauthors@list{\author@sep \csuse{##1}}%
    \author@fmt{\authors@num}{\csuse{##1t@loc@notes}}{\author@sep\csuse{##1}}%      
    }%
  \front@author@defs 
  \expandafter\docsvlist\expandafter{\author@list}%
  \par
  }
  
\def\author@list{}
  
%% <<< author setup <<<
    
%%  \corref
%%  For corresponding author
\def\corref#1{}

%%
%%  ======= EMAIL, URL =======
%%
%%  \ead, \printead
\def\email@text{{E-mail address:} }
\def\url@text{url: }
\def\fullurl@text{url: }
\def\ead@sep{;~}
\def\ead@size{%
    \@ifundefined{url}{}{\ensure@breakable@url}%
    \upshape
    \ttfamily
    }
\def\ensure@breakable@url{%
    \def\email@ead@type{email}%
    \ifx\ead@type\email@ead@type
    \else
        %%  Using \url instead of \href for autobreaking
        \def\ip@href##1##2{\url{##2}}%
        %%  Redefining tilde format in the url package (placing it higher as it was before)
        \expandafter\g@addto@macro\csname Url@OT1encSpecials\endcsname{%
            \do\~{\hbox{\m@th$\mathchar126$}}%
            }%
        %%  Ensuring \ttfamily font
        \expandafter\protect\csname url@ttstyle\endcsname
    \fi
    }

\newcounter{emailref}
\define@key{ead}{mark}[true]{%
    \usethankscounter{emailref}%
    \thankslabel{\ead@label}%
    }
\def\theemailref{\ip@getitemvalue{emailmarks}{\the\c@emailref}}
\ip@setvaluelist{emailmarks}{*,**,\textdagger,\textdaggerdbl}

\define@key{ead}{email}[true]{\def\ead@type{email}}
\define@key{ead}{url}[true]{%
    \def\@tempa{fullurl}%
    \ifx\ead@type\@tempa
    \else
        \def\ead@type{url}%
    \fi
    }
\define@key{ead}{label}{\def\ead@label{#1}}
\define@key{ead}{text}{%
    \bgroup
        \def\\{\string\break}%
        \def\break{\string\break}%
        \protected@edef\@currentlabel{#1}%
        \safelabel{\ead@label @\ead@type text}%
    \egroup
    }
\define@key{ead}{nopdflink}[true]{%
    \protected@edef\@currentlabel{nolink}%
    \safelabel{\ead@label @nopdflink}%
    }

\DeclareRobustCommand\ead[2][label= ,email]{{%
    \def\ead@type{email}%  default
    \checkead@prefix#2://\end
    \def\texttildelow{\noexpand\texttildelow}%
    \setkeys{ead}{#1}%
    \protected@edef\@currentlabel{#2}%
    \safelabel{\ead@label @\ead@type}%
    }}
\def\checkead@prefix#1://#2\end{%
    \ifx.#2.%
    \else
        \def\ead@type{fullurl}%
    \fi
    }

\newif\ifnot@ead@star
\newif\if@printead@opt
\DeclareRobustCommand{\printead}{%
    \@ifstar
        {\not@ead@starfalse\@printead}%
        {\not@ead@startrue\@printead}%
    }
\def\@printead{%
    \@ifnextchar[%
        {\@printead@opttrue\@@printead}%
        {\@printead@optfalse\@@printead[]}%
    }
\def\@@printead[#1]#2{{%
    \if@printead@opt %  []
        \def\ip@href@text{#1}%
        \not@ead@starfalse
    \fi
    \let\prev@ead@text\relax
    \let\@ead@sep\relax
    \let\ead@text\relax
    \let\ead@prefix\relax
    \def\ead@type{}%
    \@tempcnta=0%
    \let\sv@ip@href\ip@href
    \@for\ead@ref:=#2\do{%
        \advance\@tempcnta by1%
        \let\ip@href\sv@ip@href
        \@ead@sep\let\@ead@sep\ead@sep
        \@ifundefined{r@\ead@ref @nopdflink}%
            {}{\def\ip@href##1##2{##2}}%
        \@ifundefined{r@\ead@ref @email}%
            {}{\let\ead@text\email@text\def\ead@type{email}\def\ead@prefix{mailto:}}%
        \@ifundefined{r@\ead@ref @url}%
            {}{\let\ead@text\url@text\def\ead@type{url}\def\ead@prefix{http://}}%
        \@ifundefined{r@\ead@ref @fullurl}%
            {}{\let\ead@text\fullurl@text\def\ead@type{fullurl}\def\ead@prefix{}}%
        \ifx\prev@ead@text\ead@text
            \let\ead@text\relax
        \fi
        \if@printead@opt
            \ifnum\@tempcnta>1%
                \@latex@error{%
                    Command \@backslashchar printead[]{e1} could have only one parameter "e1"!%
                    }\@eha
            \fi
        \else
            \@ifundefined{r@\ead@ref @\ead@type text}{%
                \def\ip@href@text{%
                    \@ifundefined{r@\ead@ref thanks}{}{\thanksref{\ead@ref}}%
                    \saferef{\ead@ref @\ead@type}%
                    }%
                }{%
                    \def\ip@href@text{%
                        \@ifundefined{r@\ead@ref thanks}{}{\thanksref{\ead@ref}}%
                        \saferef{\ead@ref @\ead@type text}%
                        }%
                    }%
        \fi
        \ifnot@ead@star
            \ead@text
        \fi
        {%
            \ead@size
            \def\null{}%
            \@ifundefined{r@\ead@ref @\ead@type}{%
                \ip@href@text
                }{%
                    \ip@href{\ead@prefix\saferef{\ead@ref @\ead@type}}{\ip@href@text}%
                }%
            }%
        \@ifundefined{ead@text}{}{\let\prev@ead@text\ead@text}%
        }%
    }}
%%
%%  ======= ADDRESS =======
%%

\let\cny\@firstofone
\let\dept\@firstofone
\let\institution\@firstofone
\let\addr\@firstofone
\let\city\@firstofone

\let\affiliation\@gobble
\let\output@address\@gobble

\def\address@skip{6\p@ plus 1\p@ minus 1\p@}
\def\address@font{\normalsize\sffamily}
\let\safe@phantomsection\@gobble

\def\address{%
  \@ifstar
    {\address@star}
    {\@ifnextchar[{\address@optarg}{\address@noptarg}}%
  }

\def\address@optarg[#1]#2{%
  \refstepcounter{address}%
  \beg@elem
  \init@fm@element{address}{#2}%
  \csxdef{address\the\c@address @counter}{\the\c@address}%
  \temptoks=\expandafter{\@address}%
  \csxdef{address\the\c@address}{\the\temptoks}%
  \protected\csxdef{address\the\c@address t@loc@notes}{\the\t@loc@notes}%
  \@ifundefined{address@list}
    {%
      \xdef\address@list{address\the\c@address}%
    }{%
      \xappto\address@list{,address\the\c@address}%
    }%
  \label{aff.#1}%
  \csname address@end@hook\endcsname
  \ignorespaces
  }

\def\address@noptarg#1{%
  \refstepcounter{address}%
  \beg@elem
  \init@fm@element{address}{#1}%
  \csxdef{address\the\c@address @counter}{\z@}%
  \temptoks=\expandafter{\@address}%
  \csxdef{address\the\c@address}{\the\temptoks}%
  \protected\csxdef{address\the\c@address t@loc@notes}{\the\t@loc@notes}%
  \@ifundefined{address@list}
    {%
      \xdef\address@list{address\the\c@address}%
    }{%
      \xappto\address@list{,address\the\c@address}%
    }%
  \csname address@end@hook\endcsname
  \ignorespaces
  }

\def\address@star#1{%
  \beg@elem
  \init@fm@element{address}{#1}%
  \address@fmt{\m@ne}{\the\t@loc@notes}{\@address}%
  \ignorespaces
  }

\def\theaddress{\the\c@address}

\def\address@fmt#1#2#3{}
%% <<< affil setup <<<

%%  Print address by number: \printaddressnum{1}
\def\printaddressnum#1{%
    \xdef\@tmp{#1}%
    \bgroup
        \@ifundefined{address#1}{%
            \@latex@error{Error: there are no address with number '#1'!}{??}%
            }{%
                \address@font
                \ifnum#1=1%
                    \ifnum\c@author=1\relax
                        \safe@phantomsection{\addcontentsline{toc}{section}{Author's addresses}}%
                    \else
                        \safe@phantomsection{\addcontentsline{toc}{section}{Authors' addresses}}%
                    \fi
                \fi
                \begin{tabular}[t]{@{}p{\columnwidth}@{}}
                    \csname author\@tmp\endcsname. 
                    \csname address\@tmp\endcsname
                \end{tabular}%
                }%
    \egroup
    }

%%  Print all addresses:
\def\printaddresses{%
    \@tempcnta=0%
    \bgroup
        \parindent\z@
        \vskip\address@skip
        \@whilenum{\@tempcnta<\c@address}\do{%
            \advance\@tempcnta\@ne
            \vskip\address@skip
            \expandafter\printaddressnum\expandafter{\the\@tempcnta}%
            \ifnum\@tempcnta<\c@address
                \par
            \fi
            }%
    \egroup
    \null
    }

\if@load@flushend@    
    \RequirePackage[nospread]{flushend}
    %% Activating `ancient' option, which was not present in TL2010 style version
    \@ifpackagelater{flushend}{2015/04/08}{
      \@ifpackagelater{flushend}{2021/10/03}{}{\global\@ancient@balance@versiontrue}
      }{}
    %% saving discards allows to restore them to original place of last column break
    %% with \atColsBreak{\pagediscards}
    \savingvdiscards=1
    % \atColsBreak{\pagediscards}
\fi
    
%%  Invoke \printaddresses at end of document:
\AtBeginDocument{%
    \preto\enddocument{\history@fmt\printaddresses}%
    }
%%
%%  ======= DEDICATION =======
%%
\def\dedicated#1{%
    \vskip12\p@
    \bgroup
        \normalsize\itshape \centering #1\par
    \egroup
    }
%%
%%  ======= ABSTRACT =======
%%
\def\abstract@width{.8\textwidth}
\def\abstract@skip{\medskipamount}
\def\abstract@size{\normalsize}

\def\abstractname{Abstract}    
\def\abstractname@skip{}

\define@key{abstract}{language}{\set@loc@hyphenation{#1}\set@loc@abstractname{#1}}

\newbox\abstract@box
\gdef\abstract{\@ifnextchar[{\@abstract}{\@abstract[]}}
\def\@abstract[#1]{%
    \setkeys{abstract}{#1}%
    \global\setbox\abstract@box=\vbox
    \bgroup
        \hsize=\textwidth
        \ifvoid\abstract@box
        \else
            \unvbox\abstract@box
            \vskip\abstract@skip
        \fi
        \ip@normaltext
        \abstract@size
        \upshape\mdseries
        \parindent\sv@parindent
        {\centering\bfseries\textsc{\abstractname}\abstractname@skip \par}%
        \par
        \ignorespaces
    }
\def\endabstract{%
        \par
    \egroup
    }


\def\abstract@fmt{%
    \ifvoid\abstract@box
    \else
        \vskip\abstract@skip
        \unvbox\abstract@box
    \fi
    }

\def\set@loc@hyphenation#1{%
    \@ifundefined{l@#1}{}{\expandafter\language\csname l@#1\endcsname}%
    }

\def\set@loc@abstractname#1{%
    \def\abstractname@english{Abstract}%
    \def\abstractname@german{Zusammenfassung}%
    \def\abstractname@french{R\'esum\'e}%
    \def\abstractname@spanish{Resumen.}%
    \@ifundefined{abstractname@#1}%
        {\@latex@error{Missing provided language '#1` support!}{}}%
        {\edef\abstractname{\csname abstractname@#1\endcsname}}%
    }
%%
%%  ======= keyword =======
%%
% \newenvironment{keyword}[1][]{\begin{keywords}[#1]}{\end{keywords}}
\def\pf{\begin{proof}}
\def\endpf{\end{proof}}
\let\noqed\relax
\newdimen\tablewidth
\tablewidth=\textwidth

\def\keyword@skip{10\p@}
\def\keyword@width{.8\textwidth}
\def\keyword@postfix{\unskip.}
\def\keyword@font{\normalsize\upshape\mdseries}
\def\keywordname@font{\scshape}

\newbox\keyword@box
\newdimen\pre@kwd@depth

\def\keyword@AMS{AMS 2000 subject classifications:}
\def\keyword@MSC{MSC 2010 subject classifications:}
\expandafter\def\csname keyword@MSC2010\endcsname{MSC 2010 subject classifications:}
\def\keyword@KWD{keywords and phrases:}

%%  key classes
\def\keyword@class{KWD}

%%  \keyword@class-> KWD
%%  \keyword@KWD  -> AMS 2000...
\gdef\keyword{\@ifnextchar[{\@keyword}{\@keyword[class=KWD]}}
\gdef\@keyword[#1]{%
    \do@option@list{keyword}{#1}%
    \def\keyword@name{\csname keyword@\keyword@class\endcsname}%
    \let\kwd@sep\relax
    \global\setbox\keyword@box=\vbox
    \bgroup
        \ifvoid\keyword@box
        \else
            \unvbox\keyword@box
            \vskip-\pre@kwd@depth\vtop to\pre@kwd@depth{}%
        \fi
        \hsize=\textwidth
        \ip@normaltext
        \keyword@font
        \parindent\sv@parindent
        \noindent
        {\keywordname@font\keyword@name}%
        \space\hskip.1\p@
    }
\def\endkeyword{%
        \keyword@postfix
        \par
        \global\pre@kwd@depth\prevdepth
    \egroup
    }

\def\keyword@fmt{%
    \ifvoid\keyword@box
    \else
        \vskip\keyword@skip
        \unvbox\keyword@box
    \fi
    }
\newenvironment{keywords}{\keyword}{\endkeyword}
    
%%  \kwd[; ]{foo}
\def\sep{\unskip\string, }
\newif\if@firstkeywordinlist \@firstkeywordinlisttrue

\DeclareRobustCommand*\kwd{%
    \@ifnextchar[\@kwd{\@kwd[\kwd@sep]}%
    }
\def\@kwd[#1]#2{%
    \unskip#1{#2}%
    \if@firstkeywordinlist
        \addto@keywords@list{#2}%
        \@firstkeywordinlistfalse
    \else
        \addto@keywords@list{, #2}%
    \fi
    \let\kwd@sep\sep
    }

\newtoks\keywords@list
\def\addto@keywords@list#1{%
    \begingroup
        \no@harm
        \xdef\@act{%
            \global\noexpand\keywords@list{\the\keywords@list#1}%
            }%
        \@act
    \endgroup
    }
%%
%%  <<<<<<< FRONTMATTER <<<<<<<
%%
%%
%%  ======= FOOTNOTE =======
%%
\renewcommand\@makefntext[1]{%
    \@makefnmark #1%
    }
\def\@makefnmark{%
    \@textsuperscript{\normalfont\@thefnmark}%
    }

%%  The \hrule is hairline
\def\footnoterule{%
    \kern-5\p@
    \hrule width\columnwidth height .2\p@ depth\z@
    \kern 4.8\p@
    }

\RequirePackage{pifont}
    
\usepackage{textcomp}    
\DeclareTextSymbolDefault{\textasteriskcentered}{OMS}%
\DeclareTextSymbolDefault{\textdagger}{OMS}%
\DeclareTextSymbolDefault{\textdaggerdbl}{OMS}%
\DeclareTextSymbolDefault{\textsection}{OMS}%
\DeclareTextSymbolDefault{\textparagraph}{OMS}%
\DeclareTextSymbolDefault{\textbardbl}{OMS}%

%%
%%  ======= SECTIONS =======
%%
%%  Two improvements:
%%  1. if section command is defined as "inline" the '.' will be inserted after heading;
%%  2. section* will write to toc and will appear in pdf bookmarks
\def\@startsection#1#2#3#4#5#6{%
    \if@noskipsec
        \leavevmode
    \fi
    \par
    \@tempskipa #4\relax
    \@afterindenttrue
    \ifdim \@tempskipa <\z@
        \@tempskipa -\@tempskipa
        \@afterindentfalse
    \fi
    \if@nobreak
        \everypar{}%
    \else
        \addpenalty\@secpenalty\addvspace\@tempskipa
    \fi
    \@ifstar
        {\def\ssection@level{#2}\@ssect{#3}{#4}{#5}{#6}}%
        {\@dblarg{\@sect{#1}{#2}{#3}{#4}{#5}{#6}}}%
    }

%%  For hyperref
\let\aftersec@dot\relax
\def\setaftersec@dot#1{%
    \if@autosecdot
        \setbox0=\hbox{#1}%
        \ifdim\wd0>\z@
            \def\aftersec@dot{.}%
        \else
            \def\aftersec@dot{}%
        \fi
    \fi
    }

%%  Section - will add hook for the dot after section heading
\def\@sect#1#2#3#4#5#6[#7]#8{%
    \ifnum #2>\c@secnumdepth
        \let\@svsec\@empty
    \else
        \refstepcounter{#1}%
        \protected@edef\@svsec{\@seccntformat{#1}\relax}%
    \fi
    \@tempskipa #5\relax
    \ifdim \@tempskipa>\z@
        \begingroup
            #6%
            {%
                \@hangfrom{\hskip #3\relax\@svsec}%
                \interlinepenalty \@M #8\@@par
                }%
        \endgroup
        \csname #1mark\endcsname{#7}%
        \addcontentsline{toc}{#1}{%
            \ifnum #2>\c@secnumdepth
            \else
                \protect\numberline{\csname the#1\endcsname}%
            \fi
            #7%
            }%
    \else
        \setaftersec@dot{#8}%
        \def\@svsechd{%
            #6%
            {\hskip #3\relax\@svsec #8\aftersec@dot}%
            \csname #1mark\endcsname{#7}%
            \addcontentsline{toc}{#1}{%
                \ifnum #2>\c@secnumdepth
                \else
                    \protect\numberline{\csname the#1\endcsname}%
                \fi
                #7%
                }%
            }%
    \fi
    \@xsect{#5}%
    }

%%  Section* - will add hook for the dot after section heading and \contentsline
\def\@ssect#1#2#3#4#5{%
    \@tempskipa #3\relax
    \ifdim \@tempskipa>\z@
        \begingroup
            #4%
            {%
                \@hangfrom{\hskip #1}%
                \interlinepenalty \@M #5\@@par
                }%
        \endgroup
    \else
        \setaftersec@dot{#5}%
        \def\@svsechd{#4{\hskip #1\relax #5\aftersec@dot}}%
    \fi
    \ifnum\ssection@level=1%
        \phantomsection\addcontentsline{toc}{section}{#5}%
    \fi
    \@xsect{#3}%
    }

\RequirePackage{textcase}

\setcounter{secnumdepth}{4}

%%  \phantomsection is defined in hyperref
\let\phantomsection\relax

%% section, subsection etc.
\renewcommand\section{%
    \@startsection{section}{1}{\z@}%
                  {\medskipamount}%
                  {\smallskipamount}%
                  {\centering\sffamily\large\bfseries\mathversion{bold}\MakeTextUppercase}%
    }
\renewcommand\subsection{%
    \@startsection{subsection}{2}{\z@}%
                  {\smallskipamount}%
                  {\smallskipamount}%
                  {\raggedright\sffamily\large\bfseries\mathversion{bold}}%
    }
\renewcommand\subsubsection{%
    \@startsection{subsubsection}{3}{\z@}%
                  {\smallskipamount}%
                  {\smallskipamount}%
                  {\sffamily}%
    }
\renewcommand\paragraph{%
    \@startsection{paragraph}{4}{\z@}%
                  {\smallskipamount}%
                  {-1em}%
                  {\paragraph@style}%
    }
\renewcommand\subparagraph{%
    \@startsection{subparagraph}{5}{\z@}%
                  {\medskipamount}%
                  {-1em}%
                  {\subparagraph@style}%
    }
\def\paragraph@style{\normalfont\itshape}
\def\subparagraph@style{\normalfont\itshape}

%%  Format for the counter
\def\section@numbersep{.}
\def\subsection@numbersep{}
\def\subsubsection@numbersep{}
\def\paragraph@numbersep{.}
\def\subparagraph@numbersep{.}

\def\@seccntformat#1{{%
    \csname #1@prefix\endcsname\csname the#1\endcsname\csname#1@numbersep\endcsname\enspace
    }}

\let\specialsection\section
\@namedef{specialsection*}{\section*}
%%
%%  ======= APPENDIX =======
%%
\renewcommand\appendix{%
    \par
    \let\old@section\section
    \def\section{\@ifnextchar*{\@appsectionstar}{\@appsectionnostar}}%
    \def\section@prefix{\appendixname\ }%
    \def\section@numbersep{.}%
    \setcounter{section}{0}%
    \setcounter{subsection}{0}%
    \gdef\thesection{\@Alph\c@section}%
    }

\def\@appsectionstar*#1{%
    \old@section*{#1}%
    \setcounter{section}{1}%
    }

\def\@appsectionnostar#1{%
    \ifx.#1.%
        \def\section@numbersep{}%
        \old@section[\appendixname\ \thesection]{}%
    \else
        \def\section@numbersep{.}%
        \old@section{#1}%
    \fi
    }
    
    
%%
%%  ======= TABLE OF CONTENTS =======
%%
%%  For hyperref
\def\toclevel@title{0}

\newcommand*\l@title[2]{}
\newcommand*\l@author[2]{}
\newcommand*\l@doi[2]{}
\newcommand*\l@arxiv[2]{}
\newcommand*\l@jobname[2]{}
\newcommand*\l@begintocitem[2]{}
\newcommand*\l@endtocitem[2]{}
\renewcommand*\l@section{\@dottedtocline{1}{\z@}{1.5em}}

\def\enable@tableofcontents{%
    \gdef\tableofcontents{%
        \bfseries
        \@starttoc{toc}%
        }%
    }

%%  Problema: \tableofcontents reikia patraukti is fm@box,
%%  nes dingsta tarpas tarp eiluciu
\newif\if@put@toc \@put@tocfalse
\renewcommand\tableofcontents{\global\@put@toctrue}
\def\ip@tableofcontents{%
    \if@put@toc
        \nocontentsline
        \section*{\contentsname}%
        \@starttoc{toc}%
    \fi
    }
%%  Block adding to contents for the next command only:
\def\nocontentsline{%
    \let\@@addcontentsline\addcontentsline
    \ifx\hyper@anchor\@undefined
        \def\addcontentsline##1##2##3{\let\addcontentsline\@@addcontentsline}%
    \else
        \def\addcontentsline##1##2##3##4{\let\addcontentsline\@@addcontentsline}%
    \fi
    }
%%
%%  ======= LIST ENVIRONMENTS =======
%%
\parsep\z@
\topsep\smallskipamount
\partopsep\z@
\itemsep\z@
\labelsep.5em

\def\@listI{%
    \leftmargin\leftmargini
    \parsep\z@
    \topsep\smallskipamount
    \itemsep\z@
    }

\def\list@parindent{1pc}

%%  quotation
\@ifundefined{article@footnotesize}
    {\let\quotation@size\footnotesize}
    {\let\quotation@size\article@footnotesize}
\let\quotation@topsep\smallskipamount

\def\quotation{%
    \list{}{%
        \quotation@size
        \listparindent\list@parindent
        \itemindent\list@parindent
        \rightmargin\z@
        \leftmargin\list@parindent
        \partopsep\z@
        \topsep\quotation@topsep
        \parsep\z@
        }%
    \item[\Q@strut]\relax
    }
\def\endquotation{\endlist}

\def\Q@strut{%
    \leavevmode\hbox{\vrule height9\p@ depth1\p@ width\z@}%
    }

%%  quote
\@ifundefined{article@footnotesize}
    {\let\quote@size\footnotesize}
    {\let\quote@size\article@footnotesize}
\let\quote@topsep\smallskipamount

\def\quote{%
    \list{}{%
        \quote@size
        \listparindent\z@
        \itemindent\listparindent
        \rightmargin2pc%
        \leftmargin2pc%
        \partopsep\z@
        \topsep\quote@topsep
        \parsep\z@
        }%
    \item\relax
    }
\def\endquote{\endlist}
%%
%%  ======= FLOATS =======
%%
\RequirePackage{stfloats}
\fnbelowfloat
%%  Table

\def\fnum@table{\tablename~\thetable}
\setlength\belowcaptionskip{4\p@}

\renewenvironment{table}%
                 {\let\@makecaption\@maketablecaption
                  \@float{table}}%
                 {\end@float}
\renewenvironment{table*}%
                 {\let\@makecaption\@maketablecaption
                  \@dblfloat{table}}%
                 {\end@dblfloat}

%%  Various rules
\let\savehline\hline
\def\thline{\noalign{\vskip3pt}\savehline\noalign{\vskip3pt}}
\def\fhline{\noalign{\vskip1pt}\savehline\noalign{\vskip7pt}}
\def\bhline{%
    \noalign{\vskip3pt}%
    \noalign{\global\arrayrulewidth=1\p@}%
    \savehline
    \noalign{\global\arrayrulewidth=.5\p@}%
    \noalign{\vskip3pt}%
    }
\def\lhline{%
    \noalign{\vskip3pt}%
    \noalign{\global\arrayrulewidth=.3\p@}%
    \savehline
    \noalign{\global\arrayrulewidth=.5\p@}%
    \noalign{\vskip3pt}%
    }

%%  Figure
\renewcommand\figurename{Figure}

\long\def\@makecaption#1#2{%
    \vskip\abovecaptionskip
    \normalsize
    \itshape\sffamily\centering
    #1.\enskip #2\par
    \vskip\belowcaptionskip
    }

\def\@floatboxreset{%
    \reset@font
    \@setminipage
    \singlespacing
    \small
    \centering
    }

\let\@maketablecaption\@makecaption
\let\@makefigurecaption\@makecaption

%%%%%%%%%%%%%%% CONTTABLE
\newtoks\table@head

\long\def\thead#1\endthead{\global\table@head{#1}#1}
\def\contthead{\the\table@head}

\let\contpart\relax

\newenvironment{conttable}{%
  \let\footnoterule\relax
  \tablewidth=\hsize
  \@float{table}}
  {\end@float}

\newenvironment{conttable*}{%
  \let\footnoterule\relax
  \tablewidth=\textwidth
  \@dblfloat{table}}
  {\end@dblfloat}

\newenvironment{sidewaysconttable}{%
  \@rotfloat{table}}
  {\end@rotfloat}

%%%%%%%%%%%%%%% CONTCAPTION 

\RequirePackage{caption}
\def\contcaptionbraces#1#2{\gdef\contcaptionbraces@left{#1}\gdef\contcaptionbraces@right{#2}}
\contcaptionbraces()

\def\contcaption#1{%\ContinuedFloat
  \caption{\contcaptionbraces@left\textit{#1}\contcaptionbraces@right}}

\def\continued#1{\rightline{(\textit{#1})}}

\@namedef{contfigure*}{\@nameuse{figure*}}
\@namedef{endcontfigure*}{\@nameuse{endfigure*}}
\let\contfigure\figure
\let\endcontfigure\endfigure

\let\tsub\textsubscript
\let\tsup\textsuperscript

%%
%%  ======= THE BIBLIOGRAPHY =======
%%
%%% \citefix -- fixes \NAT@citex (to make a hyperlink from year component in cite command).
\def\citefix{%
    \def\NAT@citex[##1][##2]##3{%
        \NAT@sort@cites{##3}%
        \let\@citea\@empty
        \@cite{%
            \let\NAT@nm\@empty
            \let\NAT@year\@empty
            \@for\@citeb:=\NAT@cite@list\do{%
                \edef\@citeb{\expandafter\@firstofone\@citeb}%
                \if@filesw
                    \immediate\write\@auxout{\string\citation{\@citeb}}%
                \fi
                \@ifundefined{b@\@citeb\@extra@b@citeb}{%
                    \@citea{\reset@font\bfseries ?}%
                    \NAT@citeundefined
                    \PackageWarning{natbib}{Citation `\@citeb' on page \thepage \space undefined}%
                    \def\NAT@date{}%
                    }{%
                        \let\NAT@last@nm=\NAT@nm
                        \let\NAT@last@yr=\NAT@year
                        \NAT@parse{\@citeb}%
                        \ifNAT@longnames
                            \@ifundefined{bv@\@citeb\@extra@b@citeb}{%
                                \let\NAT@name=\NAT@all@names
                                \global\@namedef{bv@\@citeb\@extra@b@citeb}{}%
                                }{}%
                        \fi
                        \ifNAT@full
                            \let\NAT@nm\NAT@all@names
                        \else
                            \let\NAT@nm\NAT@name
                        \fi
                        \ifNAT@swa
                            \ifcase\NAT@ctype
                                \if\relax\NAT@date\relax
                                    \@citea\NAT@nmfmt{\NAT@nm}%
                                    \hyper@natlinkstart{\@citeb\@extra@b@citeb}%
                                    \NAT@date\hyper@natlinkend
                                \else
                                    \ifx\NAT@last@nm\NAT@nm\NAT@yrsep
                                        \ifx\NAT@last@yr\NAT@year
                                            \hyper@natlinkstart{\@citeb\@extra@b@citeb}\NAT@exlab
                                            \hyper@natlinkend
                                        \else
                                            \unskip\
                                            \hyper@natlinkstart{\@citeb\@extra@b@citeb}\NAT@date
                                            \hyper@natlinkend
                                        \fi
                                    \else
                                        \@citea\NAT@nmfmt{\NAT@nm}%
                                        \NAT@aysep\
                                        \hyper@natlinkstart{\@citeb\@extra@b@citeb}%
                                        \NAT@date\hyper@natlinkend
                                    \fi
                                \fi
                            \or
                                \@citea\hyper@natlinkstart{\@citeb\@extra@b@citeb}%
                                \NAT@nmfmt{\NAT@nm}\hyper@natlinkend
                            \or
                                \@citea\hyper@natlinkstart{\@citeb\@extra@b@citeb}%
                                \NAT@date\hyper@natlinkend
                            \or
                                \@citea\hyper@natlinkstart{\@citeb\@extra@b@citeb}%
                                \NAT@alias\hyper@natlinkend
                            \fi
                            \def\@citea{\NAT@sep\ }%
                        \else
                            \ifcase\NAT@ctype
                                \if\relax\NAT@date\relax
                                    \@citea\hyper@natlinkstart{\@citeb\@extra@b@citeb}%
                                    \NAT@nmfmt{\NAT@nm}\hyper@natlinkend
                                \else
                                    \ifx\NAT@last@nm\NAT@nm\NAT@yrsep
                                        \ifx\NAT@last@yr\NAT@year
                                            \hyper@natlinkstart{\@citeb\@extra@b@citeb}\NAT@exlab
                                            \hyper@natlinkend
                                        \else
                                            \unskip\
                                            \hyper@natlinkstart{\@citeb\@extra@b@citeb}\NAT@date
                                            \hyper@natlinkend
                                        \fi
                                    \else
                                        \@citea\NAT@nmfmt{\NAT@nm}%
                                        \ \NAT@@open
                                        \if*##1*%
                                        \else
                                            ##1\
                                        \fi
                                        \hyper@natlinkstart{\@citeb\@extra@b@citeb}%
                                        \NAT@date\hyper@natlinkend
                                    \fi
                                \fi
                            \or
                                \@citea\hyper@natlinkstart{\@citeb\@extra@b@citeb}%
                                \NAT@nmfmt{\NAT@nm}\hyper@natlinkend
                            \or
                                \@citea\hyper@natlinkstart{\@citeb\@extra@b@citeb}%
                                \NAT@date\hyper@natlinkend
                            \or
                                \@citea\hyper@natlinkstart{\@citeb\@extra@b@citeb}%
                                \NAT@alias\hyper@natlinkend
                            \fi
                            \if\relax\NAT@date\relax
                                \def\@citea{\NAT@sep\ }%
                            \else
                                \def\@citea{\NAT@@close\NAT@sep\ }%
                            \fi
                        \fi
                        }%
                }%
            \ifNAT@swa
            \else
                \if*##2*%
                \else
                    \NAT@cmt##2%
                \fi
                \if\relax\NAT@date\relax
                \else
                    \NAT@@close
                \fi
            \fi
            }{##1}{##2}%
        }%
    \let\@citex\NAT@citex
    }

\def\thebibliography@size{\footnotesize}



%%  Smart \MR
%%  Code suggested by Vilmos Prokaj <prokaj@cs.elte.hu>
%%  solves the problem when MR is in a format
%%  \MR{MR1037262 (91i:60148)}

%%  Without MR this macro removes the MR prefix if it
%%  is present unchange the argument otherwise
\def\woMR#1{\w@MR#1MR#1MR\relax}%
\def\w@MR#1MR#2MR#3\relax{#2}
\def\MRfixed{MR\woMR}%

%%  This splits MR... (...)
\def\@MR#1 #2\relax#3{%
    \href{https://mathscinet.ams.org/mathscinet-getitem?mr=#1}{\MRfixed{#3}}%
    }
\let\MR\MRfixed

\def\arxivurl#1{\href{https://arxiv.org/abs/#1}{#1}}

%%  thebibliography - structured
%%  Setting a "style" for a command:
%%  \set@bibl@cmd{bvolume}  == \def\bvolume#1{{\bvolume@style #1}}

\def\set@bibl@cmd#1{%
    \expandafter\def\csname #1\endcsname##1{{%
        \csname #1@style\endcsname##1%
        }}%
    }
\let\endbibitem\relax

\def\bibl@cmds@I{bvolume,binits,bsnm,bparticle,bsuffix,bdegs,%
  bsertitle,btitle,bbtitle,batitle,bjtitle,bctitle,bbtitle,bauthor,%
  binstitute,binstitutionaled,bissue,bnumber,bmonth,bedition,bpublisher,blocation,%
  bisbn,bissn,bfpage,blpage,byear,bcomment,bsupplement,%
  beditor,bseriesno,bconfname,bconflocation,bconfdate,%
  oauthor,oinstitute,borganization,bhowpublished,baddress}
\def\do#1{\set@bibl@cmd{#1}}
\expandafter\docsvlist\expandafter{\bibl@cmds@I}


%%  bauthor, beditor
\def\bbl@bauthor#1{%
    \csname bauthor@hook\endcsname{%
        \let\binits\@firstofone
        \let\bsnm\@firstofone
        \let\bfnm\@gobble
        \let\bparticle\@firstofone
        \let\bsuffix\@firstofone
        \bauthor@style
        #1%
        }%
    }
\def\bbl@beditor#1{{%
    \let\binits\@firstofone
    \let\bsnm\@firstofone
    \let\bfnm\@gobble
    \let\bparticle\@firstofone
    \let\bsuffix\@firstofone
    \beditor@style
    #1%
    }}
%% \bid{MR={},doi={},...}
\define@key{bid}{mr}{\bid@sep\MR{#1}}
\define@key{bid}{doi}{\bid@sep \doiurl{\doi@base #1}}
\define@key{bid}{pubmed}{}
\define@key{bid}{pii}{}
\define@key{bid}{issn}{}
\def\bbl@bid#1{\setkeys{bid}{#1}}
\ifdefined\HCode
\else
  \def\doiurl{\url}
\fi
%%  For compatibility with old bibtex style
\def\bdoi#1{\ignorespaces}

%%  common@pub@types
\def\common@pub@types{%
    \def\AND{and }%
    \let\bmisc\@firstofone
    \let\bnote\@firstofone
    \let\banumber\@firstofone
    \let\bmrnumber\MR
    \def\bid@sep{\def\bid@sep{. }}%
    \let\bid\bbl@bid
    \csname common@pub@types@hook\endcsname
    }

%%  default 
\def\AND{ and }
\def\no@editor@style{\def\beditor@style{}}
\def\oauthor@style{\aftergroup\no@editor@style\sc\selectfont}
\def\bauthor@style{\aftergroup\no@editor@style\sc\selectfont}
\def\beditor@style{\sc\selectfont}
\def\bjtitle@style{\itshape}
\def\bbtitle@style{\itshape}
\def\bsertitle@style{\itshape}
\def\bvolume@style{\bfseries}
\def\bseriesno@style{\bfseries}

\let\oauthor\bbl@bauthor

\ifdefined\HCode
\else
    %%  article
    \def\botherref{\@ifnextchar[{\@botherref}{\@botherref[]}}
    \def\@botherref[#1]{%
        \common@pub@types
        }

    %%  article
    \def\barticle{\@ifnextchar[{\@barticle}{\@barticle[]}}
    \def\@barticle[#1]{%
        \common@pub@types
        }

    %%  book, booklet,...
    \def\bbook{\@ifnextchar[{\@bbook}{\@bbook[]}}
    \def\@bbook[#1]{%
        \common@pub@types
        \let\bisbn\@gobble
        }

    %%  inproceedings, incollection
    \def\bchapter{\@ifnextchar[{\@bchapter}{\@bchapter[]}}
    \def\@bchapter[#1]{%
        \common@pub@types
        \let\bisbn\@gobble
        }
\fi

\let\betal@style\relax
\def\betal{{\betal@style et al.}}

%%  \bptok
\def\bptok#1{\ignorespaces}
\ifx\OrigBibText\undefined
    \long\def\OrigBibText#1\endOrigBibText{}
\fi

%% Acknowledgement
\define@key{acknowledgement}{title}{\def\acknowledgement@title{#1}}
\def\acknowledgement@title{Acknowledgements}
\newenvironment{acknowledgement}[1][]%
  {%
    \setkeys{acknowledgement}{#1}%
    \section*{\acknowledgement@title}%
  }{%
    \par
  }

%% Funding
\define@key{funding}{title}{\def\funding@title{#1}}
\def\funding@title{Funding}
\newenvironment{funding}[1][]%
  {%
    \setkeys{funding}{#1}%
    \section*{\funding@title}%
  }{%
    \par
  }
  
%% Funding
\define@key{supplement}{title}{\def\supplement@title{#1}}
\def\supplement@title{Supplementary Material}
\newenvironment{supplement}[1][]%
  {%
    \setkeys{supplement}{#1}%
    \section*{\supplement@title}%
  }{%
    \par
  }
  
%% esm environment for xml
\define@key{esm}{title}{\def\esm@title{#1}}
\def\esm@title{Supplementary Material}
\newenvironment{esm}[1][]%
  {%
    \def\esmdescription{\@firstofone}%
    \def\esmfiletype{\@gobble}%
    \def\esmfilename{\@gobble}%
    \setkeys{esm}{#1}%
    \def\esmtitle##1{%
      \section*{##1}%
      \esm@titlesep}%
  }{%
    \par
  }

\def\esm@titlesep{}
  
\define@key{gsponsor}{id}{\def\gsponsor@id{#1}}
\define@key{gsponsor}{sponsor-id}{\def\gsponsor@sponsorid{#1}}
\define@key{gnumber}{refid}{\def\gnumber@refid{#1}}

\def\gsponsor@fmt#1{#1}
\def\gnumber@fmt#1{#1}

\newcommand\gsponsor[2][]{%
  \bgroup
    \setkeys{gsponsor}{#1}%
    \gsponsor@fmt{#2}%
  \egroup
  }
\newcommand\gnumber[2][]{%
  \bgroup
    \setkeys{gnumber}{#1}%
    \gnumber@fmt{#2}%
  \egroup
  }

%% <<< Funding <<<

\newenvironment{backmatter}{\goodbreak\ignorespaces}{}


%%
%%  ======= REFERENCES =======
%%
%%  hyperref must be loaded after the class

%%  Interaction with hyperref
\def\ip@hyperref@settings{%
    %%  Hooks for the \thanksref, \thankstext:
    \def\thankref@hyperlink##1{%
        \edef\@tempx{##1thanks}%
        \hbox{\hyperlink{##1}{\saferef{\@tempx}}}%
        }%
    \def\thanks@hypertarget##1{%
        \smash{\raise\baselineskip\hbox{\protect\hypertarget{##1}{}}}%
        }%
    %%  Redefine \pagenumbering
    \let\pagenumbering\ip@pagenumbering
    %%  Activate href
    \let\ip@href\href
    \let\safe@phantomsection\phantomsection
    %%  Put document info
    \def\write@pdfinfo##1##2{\protected@write\@auxout{\no@harm}{\string\gdef\string##1{##2}}}%
    \@ifundefined{hy@title}{}{\pdfstringdef\@pdftitle{\hy@title}}%
    \@ifundefined{hy@author}{}{\pdfstringdef\@pdfauthor{\hy@author}}%
    \@ifundefined{hy@subject}{}{\pdfstringdef\@pdfsubject{\hy@subject}}%
    \@ifundefined{hy@keywords}{}{\pdfstringdef\@pdfkeywords{\hy@keywords}}%
    \@ifundefined{user@hy@title}{}{\global\let\@pdftitle\user@hy@title}%
    \@ifundefined{user@hy@author}{}{\global\let\@pdfauthor\user@hy@author}%
    \@ifundefined{user@hy@subject}{}{\global\let\@pdfsubject\user@hy@subject}%
    \@ifundefined{user@hy@keywords}{}{\global\let\@pdfkeywords\user@hy@keywords}%
    %%  MathSciNet:
    %%  \def\MR##1{\href{https://mathscinet.ams.org/mathscinet-getitem?mr=##1}{MR##1}}
    %%  MR with hyperef
    \def\MR##1{\@MR##1 \relax{##1}}%
    %%  \@ifundefined{Hy@SetCatcodes}{\let\MR\MRfixed}{\relax}%
    }
\def\write@pdfinfo#1#2{}

\RequirePackage{iftex}
\RequirePackage{hyperref}

\RequirePackage{url}
\urlstyle{rm}
\AfterPackage{hyperref}{%
    \hypersetup{%
        colorlinks,
        citecolor=blue,
        urlcolor=blue,
        linkcolor=blue,
        linktocpage=true
        }%
    \@ifundefined{Hy@SetCatcodes}{}{\ip@hyperref@settings}%
    %%  hyperref option 'breaklinks' does nothing for \printead links
    %%  breakurl makes proper links for all link parts
    \let\origspecial@mt\special@mt
    \def\special@mt#1#2#3{\special}%
    \ifdefined\HCode
    \else
      \ifxetex
      \else
        \RequirePackage{breakurl}%
      \fi
    \fi
    \let\special@mt\origspecial@mt
    }
%%
%%  ======= INICIALIZATION =======
%%
\singlespacing
\@twosidetrue
\pagenumbering{arabic}
\frenchspacing
\init@settings
\pagestyle{headings}

\endinput
%%
%% End of file `ipart.cls'.
