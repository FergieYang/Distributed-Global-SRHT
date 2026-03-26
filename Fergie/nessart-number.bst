%% This is file `nessart-number.bst'
%%
%% Please submit bugs or your comments to latex-support@vtex.lt
%% Start subject with a "[NESS] bibtex: " + subject
%%
%% Sorting entries is done by name and year fields!
%%
%% You are free to use this style file as you see fit, provided 
%% that you do not make changes to the file. 
%% If you DO make changes, you are required to rename this file.
%%
%% It may be distributed under the terms of the LaTeX Project Public
%% License, as described in lppl.txt in the base LaTeX distribution.
%% Either version 1.0 or, at your option, any later version.
%
%% BibTeX programming: Linas Stonys, VTeX, Lithuania, lstonys@vtex.lt 
%
% Settings:
% It's possible to pass few options to bibtex style via @settings reference. 
% To validate options, need to write that reference (@settings) citeing key (f.e: \citation{foo}) 
% to aux file. From tex file it can be done with \nocite{} command.
%
% Examples:
%   1) To make unsorted bibliography (bibs listed as cited)  
%     a) write to bib file:
%       @settings{label, options="unsort"}
%     b) write in tex file \nocite{label}
%   
%   2) To make author-year bibliography
%     a) write to bib file:
%       @settings{label, options="nameyear"}
%     b) write in tex file \nocite{label}
%   
%   3) To make alphanumerical unsorted bibliography (bibs listed as cited)  
%     a) write to bib file:
%       @settings{label, options="alpha,unsort"}
%     b) write in tex file \nocite{label}
% 
% By default, style sorts bibliography. 



ENTRY
  { address
    author
    booktitle
    chapter
    doi
    bnumber
    edition
    editor
    howpublished
    institution
    isbn
    journal
    key
    month
    note
    number
    organization
    pages   
    publisher
    school
    series
    title
    type
    volume
    year
    url
    options
    urldate
    eprint
    mrnumber
  }
  {}
  { label extra.label sort.label short.list}

INTEGERS { output.state before.all mid.sentence after.sentence
           after.block after.authors between.elements bother
           Nisbn Nmonth slen set.settings tmp month.printed}

FUNCTION {init.state.consts}
{ #0 'before.all :=
  #1 'mid.sentence :=
  #2 'after.sentence :=
  #3 'after.block :=
  #4 'after.authors :=
  #5 'between.elements :=
}

STRINGS { s t  element}
STRINGS { longest.label last.label list.string default.list}
STRINGS { v l f j b temp.str}

FUNCTION {output.nonnull}
{ 's :=
  output.state mid.sentence =
    { " " * write$ newline$}
    { output.state after.block =
        { add.period$ write$
          newline$
        }
        {
          output.state after.authors =
          { " " * write$
            newline$
          }
          { output.state between.elements =
              { ", " * write$ newline$}
              { output.state before.all =
                  'write$
                  { add.period$ " " * write$ newline$}
                if$
              }
            if$
          }
          if$
        }
      if$
      mid.sentence 'output.state :=
    }
  if$
  s
}

FUNCTION {output}
{ duplicate$ empty$
    'pop$
    'output.nonnull
  if$
}

FUNCTION {output.check}
{ 't :=
  duplicate$ empty$
    { pop$ "empty " t * " in " * cite$ * warning$ }
    'output.nonnull
  if$
}
FUNCTION {fin.entry}
{ add.period$ write$ newline$ }

FUNCTION {stupid.colon}
{ after.authors 'output.state := }


FUNCTION {insert.comma}
{ output.state before.all =
    'skip$
    { between.elements 'output.state := }
  if$
}

FUNCTION {new.sentence}
{ output.state after.block =
    'skip$
    { output.state before.all =
        'skip$
        { after.sentence 'output.state := }
      if$
    }
  if$
}

FUNCTION {not}
{   { #0 }
    { #1 }
  if$
}

FUNCTION {and}
{   'skip$
    { pop$ #0 }
  if$
}

FUNCTION {or}
{   { pop$ #1 }
    'skip$
  if$
}


FUNCTION {field.or.null}
{ duplicate$ empty$
    { pop$ "" }
    'skip$
  if$
}

FUNCTION {sort.format.month}
{ 't :=
    t #1 #2 substring$ "01" =
    t #1 #1 substring$ "1" =
    t #2 #2 substring$ "" =
    and or
    { "January" }
    { t #1 #2 substring$ "02" =
      t #1 #1 substring$ "2" =
      t #2 #2 substring$ "" =
      and or
      { "February" }
      { t #1 #2 substring$ "03" =
         t #1 #1 substring$ "3" =
         t  #2 #2 substring$ "" =
         and or
        { "March" }
        { t #1 #2 substring$ "04" =
          t #1 #1 substring$ "4" =
          or
          { "April" }
          { t #1 #2 substring$ "05" =
             t #1 #1 substring$ "5" =
             or
            { "May" }
            { t #1 #2 substring$ "06" =
              t #1 #1 substring$ "6" =
              or
              { "June" }
              { t #1 #2 substring$ "07" =
                t #1 #1 substring$ "7" =
                or
                { "July" }
                { t #1 #2 substring$ "08" =
                   t #1 #1 substring$ "8" =
                   or
                  { "August" }
                  { t #1 #2 substring$ "09" =
                    t #1 #1 substring$ "9" =
                    or
                    { "September" }
                    { t #1 #2 substring$ "10" =
                      { "October" }
                      { t #1 #2 substring$ "11" =
                        { "November" }
                        { t #1 #2 substring$ "12" =
                          { "December" }
                          { t } % No match
                        if$
                        }
                      if$
                      }
                    if$
                    }
                  if$
                  }
                if$
                }
              if$
              }
            if$
            }
          if$
          }
        if$
        }
      if$
      }
    if$
    }
  if$
 
}

INTEGERS {sk}
FUNCTION {is.num}
{ chr.to.int$
   duplicate$ "0" chr.to.int$ < not
  swap$ "9" chr.to.int$ > not and
}

FUNCTION {make.tag}
{ 't :=
  duplicate$ empty$
    { pop$ "" }
    { bother #0 =
      { 
        t "volume" =
        t "seriesno" =
        or
          {"\textbf{"  swap$ * "}" * }
          'skip$
        if$
        t "jtitle" =
        t "btitle" =
        t "sertitle" =
        or or
          {"\textit{"  swap$ * "}" * }
          'skip$
        if$
      }
      { "\b" t * "{" * swap$ * "}" * }
    if$
    }
  if$
}




FUNCTION {format.address}
{address empty$
  { "" }
  {address "address" make.tag }
  if$
}

FUNCTION {format.pub.address}
{ publisher empty$
    { "" }
    { publisher "publisher" make.tag
      address empty$
        'skip$
        {", " * format.address * add.period$}
      if$
    }
  if$
  output
}

INTEGERS { nameptr namesleft numnames }

FUNCTION {mk.tag}
{ 'temp.str :=
  duplicate$ empty$
    { pop$ "" }
    { 
      b "nothing" =
        'skip$
        { "\b" temp.str * "{" * swap$ * "}" * }
      if$
    }
  if$
}

% strip two symbols ".", "~" after dash
FUNCTION {strip.junk.after.dash}{
    t #1 #1 substring$ "." =
    t #1 #1 substring$ "~" =
    or  
      {t #2 global.max$ substring$ 't :=}
      'skip$
    if$ 
    
    t #1 #1 substring$ "." =
    t #1 #1 substring$ "~" =
    or  
      {t #2 global.max$ substring$ 't :=}
      'skip$
    if$ 
}

FUNCTION {restore.dash}{
  't :=
  "" 'element :=
  { t empty$ not }
  {  
    t #1 #5 substring$ " {qq}" =
    t #1 #5 substring$ "~{qq}" =
    or 
      {t #6 global.max$ substring$ 't := 
       strip.junk.after.dash
       element  "-" * 'element :=
      }
      {
        t #1 #4 substring$ " {q}" =
        t #1 #4 substring$ "~{q}" =
        t #1 #4 substring$ "~q.~" =
        or or
          {t #5 global.max$ substring$ 't :=
           strip.junk.after.dash
           element  "~-" * 'element :=
          }
          {element  t #1 #1 substring$ * 'element :=
            t #2 global.max$ substring$ 't :=
          }
        if$      
      }
    if$
  }
  while$
  element
}

% To hide dash we'll replace it to " {qq} " and later we'll restore it back
% otherwise bibtex will remove it when shorting initials
FUNCTION {hide.dash}{
  't :=
  "" 'element :=
  { t empty$ not }
  {  
    t #1 #3 substring$ "{-}" =
      {element  " {qq} " * 'element :=
       t #4 global.max$ substring$ 't :=
      }
      {element  t #1 #1 substring$ * 'element :=
      t #2 global.max$ substring$ 't :=
      }
    if$
  }
  while$
  element
}

FUNCTION {space.after.dot}
{'j :=
  t 'f :=
  j 't := 
  s 'j :=
  "" 'l :=
  "" 's :=
  { t empty$ not }
  {   
    t #1 #1 substring$ 's :=
    l  s * 'l := 
    s "." = 
      { 
        t #2 #1 substring$ " " =
          'skip$
           {
            l  " " * 'l := 
             t #2 #4 substring$ "{\,}" =
               { t #5 global.max$ substring$ 't := }
               'skip$
            if$ 
          } 
        if$  
      } 
      'skip$
    if$    
      t #2 global.max$ substring$ 't :=
  }
  while$
  f 't :=
  j 's :=
  l 
}

FUNCTION {fix.snm}{
  't :=
   "" 'element :=
  { t empty$ not }
  {  
    t #1 #3 substring$ "{-}" =
      {element  "-" * 'element :=
        t #3 global.max$ substring$ 't :=
       }
       {element  t #1 #1 substring$ * 'element :=}
     if$
  
    t #2 global.max$ substring$ 't :=
  }
  while$
  element
}

FUNCTION {enbrace.dash}{
  't :=
  "" 'element :=
  { t empty$ not }
  {  
    t #1 #1 substring$ "-" =
      {element  "{-}" * 'element :=}
      {element  t #1 #1 substring$ * 'element :=}
    if$
    t #2 global.max$ substring$ 't :=
  }
  while$
  element
}

FUNCTION {fix.name}{  

  enbrace.dash 's :=
  "" 'l :=
  "" 't :=

  #1 'nameptr :=
  s num.names$ 'numnames :=
  numnames 'namesleft :=
  { namesleft #0 > }
  {
     s nameptr "{vv~}{ll}{, jj}" format.name$
     s nameptr "{, ff}" format.name$ space.after.dot * 'l :=
    #1 nameptr = 
      {l 't :=}
      {t " and " * l * 't := }
    if$
    nameptr #1 + 'nameptr := 
    namesleft #1 - 'namesleft :=
  }
  while$
  t 
}

FUNCTION {name.separator}{
  namesleft #0 >
    { namesleft #1 >
        { namesleft #2 =
              { s numnames "{ll}" format.name$ "others" =
                {" "  }
                {" \AND " }
                if$
              }
              {", " }
          if$
        }
        {""}
      if$
      *
    }
    'skip$
  if$
}

FUNCTION {format.author.editor}
{ 'b :=

  bother #1 =
  b "nothing" =
  or
    'skip$
    {"b" b * 'b :=}
  if$
  
  b "editor" =
  b "beditor" =
  b "editor2" =
  b "nothing" =
  or or or
    {editor space.after.dot 's :=}
    {author space.after.dot 's :=}
  if$
  
   %% modifying name
  s fix.name  's :=
  s hide.dash 's :=
  
  "" 't :=  
 
  #1 'nameptr :=
  s num.names$ 'numnames :=
  numnames 'namesleft :=
    
  s numnames "{ll}" format.name$ 'l :=
  l  "others" =
  l  "{others}" =
  l  "et al." =
  l  "{et al.}" =
  or or or
     {#1 'tmp :=}
     {#0 'tmp :=}
  if$
    
  { namesleft #0 > }
  { s nameptr "{vv}"    format.name$ restore.dash          "particle" mk.tag  'v := 
    s nameptr "{ll}"    format.name$ fix.snm restore.dash  "snm"      mk.tag  'l :=      
    s nameptr "{jj}"    format.name$ restore.dash          "suffix"   mk.tag  'j := 
    s nameptr "{f.}"    format.name$ restore.dash          "inits"    mk.tag  'f :=     
    
     namesleft #1 =
     tmp #1 = 
     and
       {  b "bauthor" =
          b "beditor" =
          or
              {"et al."}
              {"\betal\ "}
          if$
          #1 'tmp :=
        }
        {
         b "bauthor" =
         b "beditor" =
         or
          {"\oauthor{"}
          {
             b "nothing" =
               {""}
              {
                b "editor2" =
                  {"\beditor{"}
                  {"\b" b * "{" *}
                if$
              }
            if$
          }
        if$     
        
        b "editor2" =
        b "nothing" =
        or
            {
                f empty$
                  'skip$
                  {f * " " *}     
                if$

                v empty$
                  'skip$
                  {v *  " " *}
                if$
                
                l empty$
                    'skip$
                  { l *}    
                if$     
                
                j empty$
                    'skip$
                  {" "  * j *}    
                if$
            }
            {
                v empty$
                  'skip$
                  {v * " " *}
                if$
                
                   l empty$
                    'skip$
                  { l *}    
                if$     
          
                   f empty$
                  'skip$
                  {", " * f *}    
                if$
          
                   j empty$
                    'skip$
                  {" "  * j *}    
                if$
            }
        if$

        b "nothing" =
           {""}
          {"}"}  
        if$
      
        * "" 't :=
      } 
    if$

    name.separator
    write$ 
        
    namesleft #1 >
    b "nothing" = not
    and
      { namesleft #2 =
         tmp #1 =
         and
          'skip$
          'newline$
         if$
      }
      'skip$
    if$

    nameptr #1 + 'nameptr := 
    namesleft #1 - 'namesleft :=
  }
  while$
}

FUNCTION {format.editors}
{ editor empty$
    'skip$
    {
      bother #0 =
        { author empty$
             {"editor"}
             {"nothing"}
            if$
        }
        {"editor"}
      if$  
      format.author.editor
      editor num.names$ #1 >
        { ", eds." }
        { ", ed." }
      if$
      write$ 
    }
  if$
}

FUNCTION {format.authors}
{ author empty$
    'skip$
    {"author"  format.author.editor newline$}
  if$
}

FUNCTION {make.editors}
{ editor empty$
    'skip$
    { 
      "("  output write$ 
      bother #0 =
        {author empty$
            {"editor"}
            {"nothing"}
          if$
        }
        {"editor2"}
      if$  
      format.author.editor
      editor num.names$ #1 >
        { ", eds.)"  }
        { ", ed.)"  }
      if$
    }
  if$
}

FUNCTION {cite.author.editor}
{ 's :=
  s #1 "{vv~}{ll}" format.name$
  s num.names$ duplicate$
  #2 >
    { pop$ " et~al." * }
    { #2 =
        { s #2 "{vv~}{ll}{ jj}{ f}" format.name$ "others" =
            { " et~al."  * }
            { " and " * s #2 "{vv~}{ll}" format.name$ * }
           if$
         }
         'skip$
        if$
    }
  if$
}

FUNCTION {check.auth.edit.org}
{author empty$
  {
    editor empty$
      {
        organization empty$
          {""}
          {organization}
        if$   
      }
      {editor cite.author.editor}
    if$   
  }
  {author cite.author.editor}
 if$  
}

FUNCTION {check.year}
{ year empty$
    {""}
    {year extra.label *}
  if$ 
}

INTEGERS { multiresult char.num k int}
INTEGERS {str.length count return save.num1 save.num2}
FUNCTION {is.in.list}{
  'f :=
  %%%% saveing "s" and "t"
  s 'j :=
  f 's :=
  t 'f :=  
  list.string text.length$ 'str.length :=
  list.string 't :=
%  s text.length$ 'slen :=
  int   'save.num1 :=
  count 'save.num2 :=
  #1 'int :=
  #1 'count :=
  #0 'return :=
    
    { count str.length = not 
      #0 str.length = not
      and
    }
    {
       count #1 + 'count :=
       t int #1 substring$ "," =
        {
          t #1 int substring$ s "," * =
            {
               #1 'return :=
               str.length 'count :=
            }
            'skip$
           if$
          
          int #1 + 'int :=
           t int global.max$ substring$ 't :=
          #0 'int :=
        }
        'skip$
      if$ 
      int #1 + 'int :=
    }
  while$

  save.num1 'int   :=
  save.num2 'count :=
  
  #0 return =
    { s t =
        {#1 'return :=}
        'skip$
      if$
    }
    'skip$
  if$
  %%%% returning original "s" and "t"
  j 's :=
  f 't :=  
  
  return    
}

INTEGERS {bibitem.count}

FUNCTION {print.count}
{ bibitem.count #1 + 'bibitem.count :=
  newline$
  "%%% " 
  bibitem.count int.to.str$ * write$
}
FUNCTION {output.bibitem}
{ print.count
  newline$
  "\bibitem"
  "nameyear" is.in.list
     {
       "[\protect\citeauthoryear{" * write$
           check.auth.edit.org write$ 
       "}{" write$ 
           check.year write$ 
       "}]" 
    }
    'skip$
  if$
  "alpha" is.in.list
     {
       "[" * write$
      label write$
      "]" 
    }
    'skip$
  if$

  "{" * write$  
  cite$ write$
  "}" write$

  newline$
  "" before.all 'output.state :=
}


FUNCTION {string.to.integer}
{ 't :=
  t text.length$ 'k :=
  #1 'char.num :=
  { t char.num #1 substring$ 's :=
     s is.num
     s "." =
     or 
     char.num k = not
     and
   }
   { char.num #1 + 'char.num := }
  while$
  char.num #1 - 'char.num :=
  t #1 char.num substring$   
}

 
FUNCTION {find.integer}
{ 't :=
  #0 'int :=
    { int not
      t empty$ not
      and
    }
    { t #1 #1 substring$ 's :=
      s is.num 
        {#1 'int :=}
        { t #2 global.max$ substring$ 't := }
      if$
    }
  while$
  int 
}

function{title.lowerwords}{
"a,an,and,of,the,for,in,to,over,from,on,or,so,than,that,this,as,but,with,by,at,down,up,upon,al,ale,aus,aux,d'Analyse,d'Analystes,d'Architecture,d'Etudes,d'Histoire,da,dans,de,degli,dei,del,"
"dell',dell'Accademia,dell'Istituto,dell'Universita,della,delle,dem,der,des,detta,di,die,din,dlia,do,du,e,eXpress,ed,ego,el,em,en,et,fur,i,iz,l'Academie,l'Age,l'Antiquite," *
"l'Approximation,l'Ecole,l'Histoire,l'I.S.U.P.,l'INSA,l'IREM,l'Indirizzo,l'Institut,l'Institute,l'UFR,l'Union,l'Universite,la,las,les,na,nella,per,po,si,sue,sur,ta,tes,und,van,voor,zu,zum,zur" *
}

FUNCTION {upercase.first.letters}{
  't :=
  "" 's :=
  #1 'int :=
  #1 'count :=
  list.string  'v :=
  "" 'b :=
  title.lowerwords 'list.string :=  

  t text.length$ 'slen :=
  
    {count slen = not }
    {t int #1 substring$ " " =
        {
          int #1 - 'int :=
          t #1 int substring$ 'b :=
          
          % if word not in title.lowerwords - uppercase first letter
          b is.in.list not
            { b "l" change.case$ b =
                { b "u" change.case$ "t" change.case$ 'b := }
                'skip$
              if$
            }
            'skip$
          if$
          
          int #1 + 'int :=

          s b " " *  * 's :=
          int #1 + 'int :=
           t int global.max$ substring$ 't :=
          #0 'int :=
        }
        'skip$
      if$ 
      int #1 + 'int :=
      count #1 + 'count :=
%     "count=" count int.to.str$ * " slen=" * slen int.to.str$ * top$
    }
  while$
  v 'list.string := 
  % last word is "t" and need to check it too 
  s t
  t "l" change.case$ t =
    { "u" change.case$ "t" change.case$ }
    'skip$
  if$
  *
}


FUNCTION {format.title}
{ title empty$
    { "" }
    { title }
  if$
}


FUNCTION {note.presented}
{ note #1 #9 substring$ "presented" = 
   note #1 #9 substring$ "Presented" = 
  or
}
 
FUNCTION {n.filter}
{ 't :=
  ""
  { t empty$ not }
  { t #1 #1 substring$ "," =
    t #1 #1 substring$ "\" =
    t #1 #1 substring$ "~" =
    or or
      { "" * 
        t #2 global.max$ substring$ 't := 
      }
      { t #1 #1 substring$ *
        t #2 global.max$ substring$ 't :=
      }
    if$
  }
  while$
}

FUNCTION {byear}
{ duplicate$ empty$
    { pop$ "" }
    { 't :=
       t text.length$ 'sk :=
      #5 sk =
        { t #1 #4 substring$ 's :=
          t #5 #1 substring$ 'longest.label :=
          longest.label is.num not  
            {s "year" make.tag  
              "nameyear" is.in.list 
                {longest.label * }
                'skip$
              if$
            }
            {t "year" make.tag }
          if$
        }
        { t "year" make.tag 
          "nameyear" is.in.list           
            {extra.label *}
            'skip$
          if$
        }
      if$
    }
  if$    
}

FUNCTION {format.date}
{ year empty$
    { "" }
    { year byear }
  if$
}

FUNCTION {parens}
{ "(" swap$ * ")" * }

FUNCTION {format.year}
{ year empty$
    { "" }
    { year byear parens }
  if$
}

FUNCTION {format.month}{ 
  month empty$
    {""}
    {
      type$ ".month" * is.in.list
        {
          month sort.format.month "confdate" make.tag
        }
        {""}
      if$
    }
  if$
}

FUNCTION {tie.or.space.connect}
{ duplicate$ text.length$ #3 <
    { "~" }
    { " " }
  if$
  swap$ * *
}
FUNCTION {format.volume}
{ volume empty$
    { "book" type$ =
        {number empty$
          {""}
          {number "seriesno" make.tag }
          if$
        }
        {""} 
      if$
    }
    {volume "seriesno" make.tag}
  if$
}

FUNCTION {format.art.vol}
{ 
 volume empty$
  {""}
  {volume n.filter "volume" make.tag}
  if$
  number empty$
    'skip$
    { "(" number "issue" make.tag * ")" * * }
  if$
}

FUNCTION {format.series}
{ series empty$
    {""} 
    {series "sertitle" make.tag}
  if$
}

FUNCTION {format.edition}
{ edition empty$
    { "" }
    { output.state mid.sentence =
        { edition  "l" change.case$ }
        { edition  "t" change.case$ }
      if$
      "edition" make.tag
      " ed." *
    }
  if$
}


FUNCTION {format.isbn}
{ isbn empty$
    { "" }
    { isbn "isbn" make.tag} 
  if$
}

INTEGERS {token.ascii.code}
FUNCTION {is.letter}
{ chr.to.int$ 'token.ascii.code :=
  token.ascii.code "a" chr.to.int$  #1 - < not
  token.ascii.code "z" chr.to.int$ #1 + > not and
  token.ascii.code "A" chr.to.int$ #1 - < not 
  token.ascii.code "z" chr.to.int$ #1 + > not and
  or
}


FUNCTION {multi.page.check}
{ 't :=
  #0 'multiresult :=
  "" 'l :=

    {t empty$ not}
    { t #1 #1 substring$  's :=
      
      s is.num not
        { "" l = not
            {t find.integer 
                {#1 'multiresult := }
                {#0 'multiresult := }
                if$
                "" 't :=
             }
             {
             t #2 global.max$ substring$ 't :=
             }
           if$  
        }
        { l  s * 'l :=
          t #2 global.max$ substring$ 't := 
        }
      if$
    }
  while$
  multiresult 
}
FUNCTION {clearpage}
{ 't :=
  "" 's :=
  "" 'l :=
  
    { t empty$ not }
    {
      t #1 #1 substring$ 's :=
      s is.num not
        'skip$
        { l  s * 'l :=  }
      if$
      t #2 global.max$ substring$ 't :=
    }
  while$
  l
       
}
FUNCTION {do.pages}
{'t :=
  "" 'j :=
  "" 'v :=
    {t empty$ not}
    { t #1 #1 substring$  's :=
      s is.num not
      s "," = not
      and
        { "" j = 
             {
               t #2 global.max$ substring$ 't :=
             }
            {t find.integer 
                { t clearpage  'v := }
                'skip$
                if$
                "" 't := 
             }
           if$  
        }
        { j  s * 'j :=
            t #2 global.max$ substring$ 't := 
        }
      if$
    }
  while$
  j clearpage  'j :=
  j  "fpage" make.tag 
  "cnd" is.in.list
    'skip$
    {"--"  * v "lpage" make.tag *}
  if$
}


FUNCTION {format.pages}
{ pages empty$
    { "" }
    { pages multi.page.check
        {
          pages do.pages 
        }
        {
          pages clearpage "fpage" make.tag 
        }

      if$
    }
  if$
}

FUNCTION {format.ppages}
{ pages  empty$ 
  {""}
  {pages "pages" make.tag 
     add.period$
    }
  if$
}

FUNCTION {format.chapter}
{chapter empty$
  {""}
  {"\bchapter{" chapter * "}" *   }
if$ 
}


FUNCTION {replace.tilde}
{ 't :=
  ""
    { t empty$ not }
    { t #1 #1 substring$ "~" =
            {   { t #1 #1 substring$ "~" = }
                { "\texttildelow " *
                  t #2 global.max$ substring$ 't :=
                }
              while$
            }
        { t #1 #1 substring$ *
          t #2 global.max$ substring$ 't :=
        }
      if$
    }
  while$
}


FUNCTION {format.url}
{ url empty$
    'skip$
    { type$ "article" =
        'skip$  
        { #0 bother =
            { "\url{" }
             { "\burl{" }
          if$
          url %replace.tilde 
          * "}" * 
          output  
        }
      if$
       
       urldate empty$
           'skip$
         { "Accessed " urldate * output }
       if$
    } 
  if$
}


FUNCTION {format.howpublished}
{ howpublished empty$
  {""}
    { howpublished "howpublished" make.tag }
  if$
}

FUNCTION {format.organization}
{organization empty$
    { "" }
    {organization "publisher" make.tag}
  if$
}

FUNCTION {format.org}
{organization empty$
  { "" }
  {organization "organization" make.tag add.period$}
  if$
}

FUNCTION {publisher.month.pages}{
      format.pages output
      new.sentence
      publisher empty$
        {format.organization output}
        {format.org output}
      if$

      format.pub.address
      pages empty$
          'skip$
          {insert.comma}
      if$  
}

FUNCTION {fix.doi.doimain}{
  't := % content
  t text.length$ #10 >
    {
      t #1 #10 substring$ "http://dx." = 
        {
          "https://" t #11 global.max$ substring$ * 't :=
        }
        {
          t #1 #4 substring$ "http" = 
            'skip$
            { "https://doi.org/" t * 't := }
          if$
        }
      if$ 
    }
    'skip$
  if$ 
  t 
}


% Ensuring "https://doi.org/" at begin of doi
FUNCTION {format.doi}
{ doi empty$
    {""}
    {
      "nodoi" is.in.list
        {""}
        { "\doiurl{" doi fix.doi.doimain * "}" * }
      if$
    }
  if$
}


FUNCTION {format.booktitle}
{ new.sentence
  booktitle empty$
    {""}
    {"In " booktitle "btitle" make.tag * }
  if$
}

FUNCTION {format.tr.number}
{ type empty$
    { "Technical Report" }
    'type
  if$
  
  number empty$
    { skip$ }
    { " No. \bnumber{" number * "}" * * }
  if$
}

FUNCTION {patent.number}
{ number empty$
    'skip$
    { number }
  if$
}
INTEGERS { len }

FUNCTION {chop.word}
{ 's :=
  'len :=
  s #1 len substring$ =
    { s len #1 + global.max$ substring$ }
    's
  if$
}   
 
FUNCTION {format.inpres}
{ "l" change.case$ 't :=
  "in press: " #10
  "in press. " #10
  "in press " #9 t chop.word
  chop.word
  chop.word
  #1 global.max$ substring$
}

FUNCTION {bcomment.note}
{ note empty$
    'skip$
     {note format.inpres "comment" make.tag output}
   if$
}

FUNCTION {format.thesis.type}
{ type empty$
    'skip$
    { pop$
      type "t" change.case$
    }
  if$
}


FUNCTION {item.end}
{ 
  note empty$
    'skip$
    {note.presented
        'skip$
        {new.sentence note  "comment" make.tag  output}
     if$    
    }
  if$
      
  add.period$
  format.doi output
  new.sentence
  mrnumber empty$
    'skip$
    {"\MR{" mrnumber * "}" * output}
  if$ 
  new.sentence
  eprint empty$
    'skip$
    {"\arxivurl{" eprint * "}" * output}
  if$ 
  new.sentence
  format.url 
  new.sentence
  v 'list.string :=
  fin.entry
}

FUNCTION {insert.element}
{ #0 bother =
    { element "{botherref}" * write$ newline$}
    {
      type$ "article" =
        {element "{barticle}" * write$ newline$}
        'skip$
      if$
      type$ "book" =
      type$ "proceedings" =
      type$ "manual" =
      type$ "booklet" =
      or or or 
        {element "{bbook}" * write$ newline$}
        'skip$
      if$
      
      type$ "inbook" =
        { "notnumber" last.label =
            {element "{bchapter}" * write$ newline$}
            {element "{bbook}" * write$ newline$}
          if$ 
        }
        'skip$
      if$
      
       type$ "incollection" =
       type$ "inproceedings" =
       type$ "conference" =
       or or
        {element "{bchapter}" * write$ newline$}
        'skip$
      if$
     }
  if$
}
FUNCTION {end.element}
{ "\end" 'element :=
  item.end
  insert.element
  "\endbibitem" write$ newline$
}
FUNCTION {begin.element}
{ "\begin" 'element :=
  insert.element
}

FUNCTION {article}
{ output.bibitem
  author empty$
  institution empty$ 
  editor empty$
  and and
  journal empty$ 
  or
  year empty$ 
   or 
  % above tagging rule means:
  % nead: (volume & (pages or number) ) or (doi [without volume, pages and number])
    volume empty$ not  
    pages empty$ not bnumber empty$ not or
    and
    
    volume empty$ pages empty$ bnumber empty$ and and
    doi empty$ not and  
    or
    not
  or
    {#0 'bother :=} 
     {#1 'bother :=}              
  if$ 
  begin.element
  author empty$
    { editor empty$
        'skip$
         {format.editors newline$}
      if$
    }
    { format.authors }
  if$
  format.year output
  new.sentence
  format.title "atitle" make.tag output.nonnull
  journal empty$
    'skip$
    {new.sentence journal "jtitle" make.tag  output}
  if$
  format.art.vol output
  % pages empty$
    % 'skip$
    % {insert.comma}
  % if$
   format.pages output
  end.element
}


FUNCTION {patent}
{ output.bibitem
  #0 'bother :=
  begin.element
  author empty$
  'skip$
    {format.authors}
  if$
  format.year output
  new.sentence
  format.title "title" make.tag  output.nonnull
  add.period$
  patent.number output
  insert.comma
  end.element
}

FUNCTION {book}
{ output.bibitem
  author empty$
  institution empty$ 
  organization empty$ 
  editor empty$
  and and and
  title empty$ 
   year empty$ 
   or or
    {#0 'bother :=} 
    {
      publisher empty$ address empty$ not and
        {#0 'bother :=} 
        {#1 'bother :=}
      if$
    } 
  if$ 
  begin.element
  author empty$
    { editor empty$
      'skip$
       {format.editors newline$}
      if$
    }
    { format.authors}
  if$
    format.year output
    format.title "btitle" make.tag  output
  series empty$
    {
      format.volume output
      edition empty$
        'new.sentence
        'insert.comma   
      if$
      format.edition output
      new.sentence
    }
    {
      format.edition output
      new.sentence
      format.series output
      format.volume output
      new.sentence
    }
  if$
  format.pub.address 
  end.element      
}

FUNCTION {booklet}
{output.bibitem
  author empty$
  title empty$ 
   year empty$ 
   howpublished empty$  
   or or or
    {#0 'bother :=} 
    {#1 'bother :=} 
  if$ 

  begin.element
  author empty$
    'skip$
    {format.authors}
  if$
  format.year  output
  new.sentence
  format.title "title" make.tag output
  new.sentence
  format.howpublished output
  address empty$
    'new.sentence
    'insert.comma
  if$
  format.address output
  end.element
}

FUNCTION {misc}
{output.bibitem
   #0 'bother :=
  begin.element
  author empty$
    'skip$
    {format.authors}
  if$
  format.year output
  new.sentence
  format.title "btitle" make.tag output.nonnull
  address empty$ publisher empty$ howpublished empty$ and and
    'skip$
    'new.sentence
  if$
  format.pub.address
  end.element
}

FUNCTION {inbook}
{ output.bibitem
   author empty$
  institution empty$ 
  organization empty$ 
  editor empty$
  and and and
  title empty$ 
   year empty$ 
   or or 
    {#0 'bother :=} 
    {
      publisher empty$ address empty$ not and
        {#0 'bother :=} 
        {#1 'bother :=}
      if$
    } 
  if$ 

  chapter empty$
    { "" 'last.label :=}
      { chapter string.to.integer empty$ author empty$ not and
          { "notnumber" 'last.label :=}
          { "" 'last.label :=}
        if$
      }
    if$
  
  begin.element
  author empty$
    { editor empty$
        'skip$
        {format.editors newline$}
      if$
    }
    {format.authors}
  if$
  format.year output
  new.sentence
  format.title "title" make.tag output
  format.booktitle output
  edition empty$
    'skip$
    'insert.comma
  if$   
  
  format.edition output
  series empty$
    'skip$
      {new.sentence}
  if$   
  format.series output
  format.volume output
  format.chapter output
  chapter empty$ not
  pages empty$ not
  and
      'insert.comma
      'skip$
  if$
  % format.ppages output
  format.pages output
  new.sentence
  format.pub.address 
  end.element
}

FUNCTION {incollection}
{ output.bibitem
  author empty$
  institution empty$ 
  and
  booktitle empty$ 
  title empty$
  year empty$ 
   or or or
    {#0 'bother :=} 
    {
      publisher empty$ address empty$ not and
        {#0 'bother :=} 
        {#1 'bother :=}
      if$
    } 
  if$ 
  begin.element
  author empty$
    'skip$  
    {format.authors}
  if$    
  format.year output
  new.sentence
  format.title "ctitle" make.tag output
  format.booktitle output
  format.edition output
  make.editors 
  series empty$ editor empty$
  or
      'skip$
      'new.sentence
  if$
  format.series output
  format.volume output
  format.chapter output
  chapter empty$ not
  pages empty$ not
  and
      'insert.comma
      'skip$
  if$
  % format.ppages output
  format.pages output
  format.pub.address 
  end.element
}

FUNCTION {inproceedings}
{ output.bibitem
  author empty$
  institution empty$ 
  and
  title empty$ 
  year empty$ 
   or or
    {#0 'bother :=} 
    {#1 'bother :=}
  if$ 
  begin.element
  author empty$
  'skip$
    {format.authors }
  if$
  format.year output
  new.sentence
  format.title "title" make.tag output.nonnull
  format.booktitle output
  make.editors 
  series empty$
    'skip$
    'new.sentence
  if$
  format.series output
  format.volume output
  % format.ppages output
  format.pages output
    new.sentence
  publisher empty$
      {format.organization output}
        {format.org output}
    if$
    format.pub.address 
  end.element
}

FUNCTION {conference} { inproceedings }

FUNCTION {manual}
{ output.bibitem
  author empty$
  organization empty$ 
  title empty$ 
   year empty$ 
   address  empty$ 
   or or or or
    {#0 'bother :=} 
    {#1 'bother :=} 
  if$ 
  begin.element

  author empty$
    { organization empty$
        'skip$
        { organization "publisher" make.tag output.nonnull
          stupid.colon% add.period$
        }
      if$
    }
    { format.authors}
  if$
  format.year output
  new.sentence
  title empty$
    'skip$
    {format.title "title" make.tag output}
   if$
  edition empty$
     {new.sentence}
     {insert.comma}
  if$
  format.edition output
  edition empty$
    'skip$
     {new.sentence}
  if$
   author empty$ organization empty$
   or
    'skip$
    { organization "publisher" make.tag output.nonnull
      insert.comma
    }
  if$
  address empty$
    'skip$
     {address "location" make.tag output}
  if$
  end.element
}


FUNCTION {phdthesis}
{output.bibitem
  #0 'bother :=
  begin.element
  author empty$
  'skip$
    {format.authors}
  if$
  format.title "title" make.tag output.nonnull
  format.year output
  new.sentence
  type$ "mastersthesis"  =
    {"Master's thesis" format.thesis.type output.nonnull}
    {"PhD thesis" format.thesis.type output.nonnull}
  if$   
  school empty$
    'skip$
     'insert.comma 
  if$
  school "school" output.check
  address empty$
    'skip$
     'insert.comma 
  if$
  address output
  end.element
}
FUNCTION {mastersthesis}{phdthesis}


FUNCTION {proceedings}
{ output.bibitem
  author empty$
  institution empty$ 
  organization empty$ 
  editor empty$
  and and and
  title empty$ 
  year empty$ 
   or or 
    {#0 'bother :=} 
    {
      publisher empty$ address empty$ not and
        {#0 'bother :=} 
        {#1 'bother :=}
      if$
    } 
  if$ 

  begin.element
  author empty$
     { editor empty$
       {organization "institutionaled" make.tag "organization" output.check }
    { format.editors newline$}
      if$ 
    }
    { format.authors }
  if$
  format.year output
  new.sentence
  format.title "btitle" make.tag output
  series empty$
    'skip$
    {new.sentence format.series output}
  if$
  volume empty$
    'skip$
    'insert.comma  
  if$  
  format.volume output
  publisher.month.pages
  end.element
}

FUNCTION {techreport}
{ output.bibitem
  #0 'bother :=
  begin.element
  author empty$
    'skip$
    {format.authors}
  if$
  format.year output  
  new.sentence
  format.title "title" make.tag output.nonnull
  new.sentence
  format.tr.number output.nonnull
  institution empty$
    'skip$
     'insert.comma 
  if$
  institution "institution" output.check
  address empty$
    'skip$
    'insert.comma 
  if$
  address output
  end.element
}

FUNCTION {unpublished}
{ output.bibitem
  #0 'bother :=

  begin.element
  author empty$
  'skip$
    {format.authors}
  if$
  format.year output
  new.sentence
  format.title "title" make.tag output.nonnull
  end.element
}

FUNCTION {settings} {}
FUNCTION {default.type} { unpublished }

MACRO {jan} {"January"}
MACRO {feb} {"February"}
MACRO {mar} {"March"}
MACRO {apr} {"April"}
MACRO {may} {"May"}
MACRO {jun} {"June"}
MACRO {jul} {"July"}
MACRO {aug} {"August"}
MACRO {sep} {"September"}
MACRO {oct} {"October"}
MACRO {nov} {"November"}
MACRO {dec} {"December"}

READ

FUNCTION {sortify}
{ purify$
  "l" change.case$
}

INTEGERS { et.al.char.used }

FUNCTION {initialize.et.al.char.used}
{ #0 'et.al.char.used :=
}

%%%%% setting default options
FUNCTION {set.default.opt}
{"sor" 'list.string :=}

EXECUTE {set.default.opt}

FUNCTION {assign.opt}{
  % First need to set options
  "settings" type$ =
    {options empty$
        'skip$
        {options 'list.string :=}
       if$
    }
    'skip$
  if$
}

ITERATE {assign.opt}
EXECUTE {initialize.et.al.char.used}

FUNCTION {alpha.format.lab.names}
{ 's :=
  s num.names$ 'numnames :=
  numnames #1 >
    { numnames #4 >
        { #3 'namesleft := }
        { numnames 'namesleft := }
      if$
      #1 'nameptr :=
      ""
      { namesleft #0 > }
      { nameptr numnames =
          { s nameptr "{ff }{vv }{ll}{ jj}" format.name$ "others" =
        { "+" *
          #1 'et.al.char.used :=
        }
        { s nameptr "{v{}}{l{}}" format.name$ * }
            if$
          }
          { s nameptr "{v{}}{l{}}" format.name$ * }
        if$
        nameptr #1 + 'nameptr :=
        namesleft #1 - 'namesleft :=
      }
      while$
      numnames #4 >
         { "+" *
          #1 'et.al.char.used :=
        }
        'skip$
      if$
    }
    { s #1 "{v{}}{l{}}" format.name$
      duplicate$ text.length$ #2 <
        { pop$ s #1 "{ll}" format.name$ #3 text.prefix$ }
        'skip$
      if$
    }
  if$
}

FUNCTION {standard.format.lab.names}
{ 's :=
  s #1 "{vv~}{ll}" format.name$
  s num.names$ duplicate$
  #2 >
    { pop$ " et~al." * }
    { #2 <
        'skip$
        { s #2 "{ff }{vv }{ll}{ jj}" format.name$ "others" =
            { " et~al." * }
            { " and " * s #2 "{vv~}{ll}" format.name$ * }
          if$
        }
      if$
    }
  if$
}
FUNCTION {format.lab.names}{
  "alpha" is.in.list
    'alpha.format.lab.names
    'standard.format.lab.names
  if$
}

FUNCTION {author.key.label}
{ author empty$
    { key empty$
        { cite$ #1 #3 substring$ }
        { key #3 text.prefix$ }
      if$
    }
    { author format.lab.names }
  if$
}

FUNCTION {author.editor.key.label}
{ author empty$
    { editor empty$
        { key empty$
            { "zzz" }
            { key #3 text.prefix$ }
          if$
        }
        { editor format.lab.names }
      if$
    }
    { author format.lab.names }
  if$
}

FUNCTION {author.key.organization.label}
{ author empty$
    { key empty$
        { organization empty$
            { "zzz" }
            { "The " #4 organization chop.word #3 text.prefix$ }
          if$
        }
        { key #3 text.prefix$ }
      if$
    }
    { author format.lab.names }
  if$
}

FUNCTION {editor.key.organization.label}
{ editor empty$
    { key empty$
        { organization empty$
            { "zzz" }
            { "The " #4 organization chop.word #3 text.prefix$ }
          if$
        }
        { key #3 text.prefix$ }
      if$
    }
    { editor format.lab.names }
  if$
}

FUNCTION {calc.short.authors}
{ type$ "book" =
  type$ "inbook" =
  or
    'author.editor.key.label
    { type$ "proceedings" =
        'editor.key.organization.label
        { type$ "manual" =
            'author.key.organization.label
            'author.key.label
          if$
        }
      if$
    }
  if$
  'short.list :=
}

FUNCTION {calc.label}
{ calc.short.authors
  short.list

  "alpha" is.in.list
  {
    duplicate$
    year field.or.null purify$ #-1 #2 substring$
    *
    'label :=
    year field.or.null purify$ #-1 #4 substring$
    *
    sortify 'sort.label :=
  }
  {
    "("
    *
    year duplicate$ empty$
    short.list key field.or.null = or
      { pop$ "" }
      'skip$
    if$
    *
    'label :=
    label 'sort.label :=
  } 
  if$
}

FUNCTION {sort.format.names}
{ 's :=
  #1 'nameptr :=
  ""
  s num.names$ 'numnames :=
  numnames 'namesleft :=
    { namesleft #0 > }
    { nameptr #1 >
        { "   " * 
          "alpha" is.in.list
            'skip$
            {numnames int.to.str$ * "   " *}
          if$
        }
        'skip$
      if$
      s nameptr "{vv{ } }{ll{ }}{  ff{ }}{  jj{ }}" format.name$ 't :=
      nameptr numnames = 
      t "others" = 
      and
        { "et al" * }
        { t sortify * }
      if$
      #3 numnames  <
      "alpha" is.in.list not
      and
        {#0 'namesleft :=
             " zzz   " *
        }    
        {
           nameptr #1 + 'nameptr :=
          namesleft #1 - 'namesleft :=
        }
      if$   
    }
  while$
}

FUNCTION {sort.format.title}
{ 't :=
  "A " #2
  "An " #3
  "The " #4 t chop.word
  chop.word
  chop.word
  sortify
  #1 global.max$ substring$
}

FUNCTION {author.sort}
{ author empty$
    { key empty$
        { "to sort, need author or key in " cite$ * warning$
          "zzz"
        }
        { key sortify }
      if$
    }
    { author sort.format.names }
  if$
}

FUNCTION {author.editor.sort}
{ author empty$
    { editor empty$
        { key empty$
            { "to sort, need author, editor, or key in " cite$ * warning$
              "z"
            }
            { key sortify }
          if$
        }
        { editor sort.format.names }
      if$
    }
    { author sort.format.names }
  if$
}

FUNCTION {author.organization.sort}
{ author empty$
    { organization empty$
        { key empty$
            { "to sort, need author, organization, or key in " cite$ * warning$
              "z"
            }
            { key sortify }
          if$
        }
        { "The " #4 organization chop.word sortify }
      if$
    }
    { author sort.format.names }
  if$
}

FUNCTION {editor.organization.sort}
{ editor empty$
    { organization empty$
        { key empty$
            { "to sort, need editor, organization, or key in " cite$ * warning$
              "z"
            }
            { key sortify }
          if$
        }
        { "The " #4 organization chop.word sortify }
      if$
    }
    { editor sort.format.names }
  if$
}

FUNCTION {presort}
{ calc.label
  sort.label sortify
  "    "
  *
  type$ "book" =
  type$ "inbook" =
  or
     {author empty$
       'editor.organization.sort
      'author.organization.sort
     if$
    }
    { type$ "proceedings" =
      type$ "incollection" =
       or
        {author empty$
            'editor.organization.sort
            'author.organization.sort
        if$
        }
        { type$ "manual" =
            'author.organization.sort
            'author.sort
          if$
        }
      if$
    }
  if$
  "    "
  *
  year field.or.null sortify
  *
  "    "
  *
  title field.or.null
  sort.format.title
  *
  #1 entry.max$ substring$ 'sort.label :=
  sort.label *
  #1 entry.max$ substring$ 'sort.key$ :=
  
  "settings" type$ =
    {"aaa"}
    {"unsort" is.in.list
      {"bb"}
      {"alpha" is.in.list
        {sort.key$}
        {sort.label}
        if$  
       }
     if$
    }
    if$
    
  'sort.key$ :=
}


ITERATE {presort}

SORT

INTEGERS { longest.label.width last.extra.num number.label }

FUNCTION {initialize.longest.label}
{ "" 'longest.label :=
  #0 int.to.chr$ 'last.label :=
  "" 'j :=
  #0 'longest.label.width :=
  #0 'last.extra.num :=
  #0 'number.label :=
}

FUNCTION {forward.pass}
{ last.label label =
    { last.extra.num #1 + 'last.extra.num :=
      last.extra.num int.to.chr$ 'extra.label :=
    }
    { "a" chr.to.int$ 'last.extra.num :=
      "" 'extra.label :=
      label 'last.label :=
    }
  if$
  number.label #1 + 'number.label :=
}

FUNCTION {reverse.pass}
{ j "b" =
    { "a" 'extra.label := }
    'skip$
  if$
  extra.label 'j :=
  label extra.label * 'label :=
}

EXECUTE {initialize.longest.label}

ITERATE {forward.pass}

REVERSE {reverse.pass}

FUNCTION {write.preambule}
{   "\begin{thebibliography}{" number.label int.to.str$ * "}" * write$ newline$
  "% BibTex style file: nessart-number.bst, 2022-10-06" write$ newline$
  "\ifx \bisbn   \undefined \def \bisbn  #1{ISBN #1}\fi" write$ newline$
  "\ifx \binits  \undefined \def \binits#1{#1}\fi" write$ newline$
  "\ifx \bauthor  \undefined \def \bauthor#1{#1}\fi" write$ newline$
  "\ifx \batitle  \undefined \def \batitle#1{#1}\fi" write$ newline$
  "\ifx \bjtitle  \undefined \def \bjtitle#1{\textit{#1}}\fi" write$ newline$ 
  "\ifx \bvolume  \undefined \def \bvolume#1{\textbf{#1}}\fi" write$ newline$
  "\ifx \byear  \undefined \def \byear#1{#1}\fi" write$ newline$
  "\ifx \bissue  \undefined \def \bissue#1{#1}\fi" write$ newline$
  "\ifx \bfpage  \undefined \def \bfpage#1{#1}\fi" write$ newline$
  "\ifx \blpage  \undefined \def \blpage #1{#1}\fi" write$ newline$
  "\ifx \burl  \undefined \def \burl#1{\textsf{#1}}\fi" write$ newline$
  "\ifx \doiurl  \undefined \def \doiurl#1{\textsf{#1}}\fi" write$ newline$
  "\ifx \betal  \undefined \def \betal{\textit{et al.}}\fi" write$ newline$
  "\ifx \binstitute  \undefined \def \binstitute#1{#1}\fi" write$ newline$
  "\ifx \binstitutionaled  \undefined \def \binstitutionaled#1{#1}\fi" write$ newline$
  "\ifx \bctitle  \undefined \def \bctitle#1{#1}\fi" write$ newline$
  "\ifx \beditor  \undefined \def \beditor#1{#1}\fi" write$ newline$
  "\ifx \bpublisher  \undefined \def \bpublisher#1{#1}\fi" write$ newline$
  "\ifx \bbtitle  \undefined \def \bbtitle#1{\textit{#1}}\fi" write$ newline$
  "\ifx \bedition  \undefined \def \bedition#1{#1}\fi" write$ newline$
  "\ifx \bseriesno  \undefined \def \bseriesno#1{\textbf{#1}}\fi" write$ newline$
  "\ifx \blocation  \undefined \def \blocation#1{#1}\fi" write$ newline$
  "\ifx \bsertitle  \undefined \def \bsertitle#1{\textit{#1}}\fi" write$ newline$
  "\ifx \bsnm \undefined \def \bsnm#1{#1}\fi" write$ newline$
  "\ifx \bsuffix \undefined \def \bsuffix#1{#1}\fi" write$ newline$
  "\ifx \bparticle \undefined \def \bparticle#1{#1}\fi" write$ newline$
  "\ifx \barticle \undefined \def \barticle{}\fi" write$ newline$
}


FUNCTION {begin.bib}
{ preamble$ empty$
    'skip$
    { preamble$ write$ newline$ }
  if$
  write.preambule
  "\ifx \borganization \undefined \def \borganization #1{#1} \fi" write$ newline$
  "\ifx \bhowpublished \undefined \def \bhowpublished #1{#1} \fi" write$ newline$
  "\ifx \bconfdate \undefined \def \bconfdate #1{#1} \fi" write$ newline$
  "\ifx \botherref \undefined \def \botherref #1{#1} \fi" write$ newline$
  "\ifx \url \undefined \def \url#1{\textsf{#1}} \fi" write$ newline$
  "\ifx \bchapter \undefined \def \bchapter{} \fi" write$ newline$
  "\ifx \bbook \undefined \def \bbook{} \fi" write$ newline$
  "\ifx \bcomment \undefined \def \bcomment#1{#1} \fi" write$ newline$
  "\ifx \oauthor \undefined \def \oauthor#1{#1} \fi" write$ newline$
  "\ifx \citeauthoryear \undefined \def \citeauthoryear#1{#1} \fi" write$ newline$
  "\ifx \endbibitem  \undefined \def \endbibitem {}\fi" write$ newline$
  "\ifx \bconflocation  \undefined \def \bconflocation#1{#1} \fi" write$ newline$
  "\ifx \arxivurl  \undefined \def \arxivurl#1{\textsf{#1}} \fi" write$ newline$
  "\csname PreBibitemsHook\endcsname" write$ newline$
}
EXECUTE {begin.bib}

EXECUTE {init.state.consts}

ITERATE {call.type$}

FUNCTION {end.bib}
{ newline$
 "\end{thebibliography}" write$ newline$
}

EXECUTE {end.bib}




