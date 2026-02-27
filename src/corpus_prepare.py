import random

corpusSplitString = '@\n'
maxPoemLength = 10000
symbolCountThreshold = 100
authorCountThreshold = 20
START_CHAR = '{'
END_CHAR = '}'
UNK_CHAR = '@'
PAD_CHAR = '|'

def splitSentCorpus(fullSentCorpus, testFraction = 0.1):
    random.seed(42)
    random.shuffle(fullSentCorpus)
    testCount = int(len(fullSentCorpus) * testFraction)
    testSentCorpus = fullSentCorpus[:testCount]
    trainSentCorpus = fullSentCorpus[testCount:]
    return testSentCorpus, trainSentCorpus

def getAlphabetAuthors(corpus):
    symbols={}
    authors={}
    for s in corpus:
        if len(s) > 0:
            n=s.find('\n')
            aut = s[:n]
            if aut in authors: authors[aut] += 1
            else: authors[aut] = 1
            poem = s[n+1:]
            for c in poem:
                if c in symbols: symbols[c] += 1
                else: symbols[c]=1
    return symbols, authors

def prepare_data(corpusFileName, startChar=START_CHAR, endChar=END_CHAR, unkChar=UNK_CHAR, padChar=PAD_CHAR):
    file = open(corpusFileName,'r')
    poems = file.read().split(corpusSplitString)
    symbols, authors = getAlphabetAuthors(poems)
    
    assert startChar not in symbols and endChar not in symbols and unkChar not in symbols and padChar not in symbols
    charset = [startChar,endChar,unkChar,padChar] + [c for c in sorted(symbols) if symbols[c] > symbolCountThreshold]
    char2id = { c:i for i,c in enumerate(charset)}
    authset = [a for a in sorted(authors) if authors[a] > authorCountThreshold]
    auth2id = { a:i for i,a in enumerate(authset)}
    
    corpus = []
    for i,s in enumerate(poems):
        if len(s) > 0:
            n=s.find('\n')
            aut = s[:n]
            poem = s[n+1:]
            corpus.append( (aut,[startChar] + [ poem[i] for i in range(min(len(poem),maxPoemLength)) ] + [endChar]) )

    testCorpus, trainCorpus  = splitSentCorpus(corpus, testFraction = 0.01)
    print('Corpus loading completed.')
    return testCorpus, trainCorpus, char2id, auth2id

def load_corpus(corpusFileName):
	with open(corpusFileName, 'r') as f:
		corpus = f.read().split('@')
	
	charset = set([START_CHAR, END_CHAR, UNK_CHAR, PAD_CHAR])
	for poem in corpus:
		poem = poem[1:-1]
		for char in poem:
			charset.add(char)

	charset = list(charset)
	charset.sort()
	char2id = { c:i for i,c in enumerate(charset)}

	result_corpus = [[poem[i] for i in range(min(len(poem),maxPoemLength))] for poem in corpus]
	return result_corpus, char2id
