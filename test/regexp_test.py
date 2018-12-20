import string
import re


def check_naive(mystring):

    ret = ''
    for c in mystring :
         if c in string.printable:
             ret = ret + c

    return ret





text = 'for (int i = 1; i <= size; i++)     /*주석*/ , abc="someText"      { ' \
    'char ch = (char) ((Math.random() * 26) + 65);' \
       '장재훈  ' \
       'result.append(ch);}'


print(check_naive(text))


text = '   a      abc   def  e'
print(re.sub('  +',' ',text))




