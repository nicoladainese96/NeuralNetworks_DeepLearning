import re
import numpy as np
import os

def get_sentences(text, sep='.\n'):
    sentences = re.split(sep, text)
    sequences = []
    for s in sentences:
        sequences.append(s+' .\n')
    return sequences

def get_words(sentence, sep=' '):
    words = re.split(sep, sentence)
    filtered_words = []
    for w in words:
        if w != '':
            filtered_words.append(w)
        else:
            pass
    return filtered_words

def join_sequences(list_of_seq):
    text = ''
    for s in list_of_seq:
        text += s+' '
    text = text.replace('  ',' ')
    return text

def get_vocabularies(text):
    words = get_words(text)
    unique_words = list(set(words))
    word2index = {}
    index2word = {}
    word2index['<PAD>'] = 0
    index2word[0] = '<PAD>'
    for i,w in enumerate(unique_words):
        if w == '.\n':
            word2index[w] = i+1
            index2word[i+1] = '.'
        else:
            word2index[w] = i+1
            index2word[i+1] = w
    return word2index, index2word

def numerical_encoder(sentence, word2index):
    num_sentence = []
    words = get_words(sentence)
    for w in words:
        index = word2index[w]
        num_sentence.append(index)
    return np.array(num_sentence)

def numerical_decoder(num_sentence, index2word):
    word_sentence = []
    for n in num_sentence:
        w = index2word[n]
        word_sentence.append(w)
    return word_sentence
    
def get_corpus(filepath, debug = True):
    print('\n\n'+'='*100)
    print("Preprocessing "+filepath, '\n\n')
    with open(filepath, 'r') as f:
        original_text = f.read()
    #text = re.split('\n{5}', original_text)
    text = original_text.split(sep='End of the Project Gutenberg EBook')[0]
    text = text.split(sep='CHAPTER')
    #print("len(text) after 'CHAPTER' splitting: ", len(text), '\n')
    chapters = []
    for i in range(len(text)):
        if len(text[i])>2000:
            chapters.append(text[i])
    sentences = []
    for c in chapters: 
        lengths = np.array([len(x) for x in re.split('\n{3}', c)])
        mask = (lengths > 10)
        if debug:
            print("Corpus lengths: ", lengths)
        splitted_chap = re.split('\n{3}', c)
        corpus = ''
        for i in range(len(splitted_chap)):
            if mask[i] == 1:
                corpus += ' \n ' + splitted_chap[i]
            else:
                continue
        sentences += re.split('\n{2}', corpus)
        
        if debug:
            print('Chapter length: ', len(c))
            print('Corpus length: ', len(corpus))
            print('Num. sentences: ', len(sentences))
            print('Char. for seq. : ', len(corpus)/len(sentences))
            print('Chapter init: ', corpus[:100], '\n')
        index = np.random.choice(np.arange(len(sentences)))
        if debug:
            print('Random sentence/paragraph: \n', sentences[index], '\n')
        
        
    full_text = ''
    for s in sentences:
        s = re.sub('\n', ' ', s)
        full_text += s
    full_text = full_text.lower()
    return full_text
   
def get_corpus_lady_susan(filepath, debug = True):
    print('\n\n'+'='*100)
    print("Preprocessing "+filepath, '\n\n')
    with open(filepath, 'r') as f:
        original_text = f.read()
    text = original_text.split(sep='End of the Project Gutenberg EBook')[0]
    text = re.split('\n{5}', text)
    if debug:
        print("len(text) after 5 newline splitting: ", len(text), '\n')
    chapters = []
    for i in range(len(text)):
        if len(text[i])>2000:
            chapters.append(text[i])
    sentences = []
    for c in chapters: 
        lengths = np.array([len(x) for x in re.split('\n{3}', c)])
        mask = (lengths > 50)
        if debug:
            print("Corpus lengths: ", lengths)
        splitted_chap = re.split('\n{3}', c)
        corpus = ''
        for i in range(len(splitted_chap)):
            if mask[i] == 1:
                corpus += ' \n ' + splitted_chap[i]
            else:
                continue
        sentences += re.split('\n{2}', corpus)
        
        if debug:
            print('Chapter length: ', len(c))
            print('Corpus length: ', len(corpus))
            print('Num. sentences: ', len(sentences))
            print('Char. for seq. : ', len(corpus)/len(sentences))
            print('Chapter init: ', corpus[:100], '\n')
        index = np.random.choice(np.arange(len(sentences)))
        if debug:
            print('Random sentence/paragraph: \n', sentences[index], '\n')
        
        
    full_text = ''
    for s in sentences:
        s = re.sub('\n', ' ', s)
        full_text += s
    full_text = full_text.lower()
    return full_text

def get_corpus_persuasion(filepath, debug = True):
    print('\n\n'+'='*100)
    print("Preprocessing "+filepath, '\n\n')
    with open(filepath, 'r') as f:
        original_text = f.read()
    text = original_text.split(sep='End of the Project Gutenberg EBook')[0]
    text = text.split(sep='Chapter')
    #text = re.split('\n{5}', text)
    if debug:
        print("len(text) after 'Chapter' splitting: ", len(text), '\n')
    chapters = []
    for i in range(len(text)):
        if len(text[i])>2000:
            chapters.append(text[i])
    sentences = []
    for c in chapters: 
        lengths = np.array([len(x) for x in re.split('\n{3}', c)])
        mask = (lengths > 50)
        if debug:
            print("Corpus lengths: ", lengths)
        splitted_chap = re.split('\n{3}', c)
        corpus = ''
        for i in range(len(splitted_chap)):
            if mask[i] == 1:
                corpus += ' \n ' + splitted_chap[i]
            else:
                continue
        sentences += re.split('\n{2}', corpus)
        
        if debug:
            print('Chapter length: ', len(c))
            print('Corpus length: ', len(corpus))
            print('Num. sentences: ', len(sentences))
            print('Char. for seq. : ', len(corpus)/len(sentences))
            print('Chapter init: ', corpus[:100], '\n')
        index = np.random.choice(np.arange(len(sentences)))
        if debug:
            print('Random sentence/paragraph: \n', sentences[index], '\n')
        
        
    full_text = ''
    for s in sentences:
        s = re.sub('\n', ' ', s)
        full_text += s
    full_text = full_text.lower()
    return full_text


def get_corpus_pride_prejudice(filepath, debug = True):
    print('\n\n'+'='*100)
    print("Preprocessing (ad hoc) "+filepath)
    with open(filepath, 'r') as f:
        original_text = f.read()
    text = re.split('\n{5}', original_text)
    #print("len(text) after 5 newline splitting: ", len(text), '\n')
    chapters = []
    for i in range(len(text)):
        if len(text[i])>2000:
            #print('Chapter length: ', len(text[i]), '\nChapter init: ', text[i][:100])
            chapters.append(text[i])
    sentences = []
    for c in chapters[:-1]: # drop last part of the text (added by Gutenberg project)
        corpus = re.split('\n{3}', c)[1:][0]
        sentences += re.split('\n{2}', corpus)
        
    full_text = ''
    for s in sentences:
        s = re.sub('\n', ' ', s)
        full_text += s
    full_text = full_text.lower()
    return full_text
    
def austen_preprocessing(filepath='austen.txt', debug=False, verbose=False):
    """
    Preprocessing
    -------------
    Read text, drop initial and final part added by Gutenberg project, split in chapters and remove headings.
    Map upper case characters to lowercase ones.
    Remove all non-alphabetic characters except for point, comma and newline.
    Replace "mr." and "mrs." with "mr" and "mrs" (needed for the correct subsequent processing).
    Replace "à" and "ê" with "a" and "e".
    Replace all the points with "./n" (useful for futher splitting in sequences).
    Add a space in front of all commas (they must be interpreted as words in the encoding).
    Remove double spaces.
    
    Return
    ------
    
    
    """
    
    files = os.listdir('Books')
    print(files)
    
    full_text = ''
    for f in files:
        if f == 'pride_prejudice.txt':
            full_text += ' '+get_corpus_pride_prejudice('Books/'+f)
        elif f == 'lady_susan.txt':
            full_text += ' '+get_corpus_lady_susan('Books/'+f, debug = debug)
        elif f == 'persuasion.txt':
            full_text += ' '+get_corpus_persuasion('Books/'+f, debug = debug)
        else:
            full_text += ' '+get_corpus('Books/'+f, debug = debug)
    
    special_chars = ['!','?','&','(',')','*','-','_',':',';','"','\'',
                     '£','[',']', '“', '”',
                     '1','2','3','4','5','6','7','8','9','0']
    for x in special_chars:
        full_text = full_text.replace(x,' ')
    full_text = full_text.replace('mr.','mr')
    full_text = full_text.replace('mrs.','mrs')
    full_text = re.sub('à', 'a', full_text)
    full_text = re.sub('ê', 'e', full_text)
    full_text = re.sub('é', 'e', full_text)
    full_text = re.sub(r'[.]','.\n', full_text)
    full_text = full_text.replace(',',' ,')
    full_text = full_text.replace('  ',' ')


    if verbose:
        alphabet = list(set(full_text))
        alphabet.sort()
        print('Found letters:', alphabet)
        
    sentences = get_sentences(full_text)
    full_text = join_sequences(sentences) # ".\n" -> " .\n" so that it becomes a separable word
    word2index, index2word = get_vocabularies(full_text)
    num_sentences = []
    for i,s in enumerate(sentences):
        num_s = numerical_encoder(s,word2index)
        num_sentences.append(num_s)
        if i==0 and verbose == True:
            print("Original sentence : \n")
            print(s)
            print("Retrieved sentence : \n")
            print(numerical_decoder(num_s, index2word))
    
    dataset =dict(word2index=word2index, index2word=index2word, num_sentences=num_sentences )
    return dataset