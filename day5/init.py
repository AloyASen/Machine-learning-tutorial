from nltk.tokenize import word_tokenize, sent_tokenize
#tokenizing
# word tokenizers ... sentence tokenizers

EXAMPLE_TEXT = "Hello Mr. Smith, how are you doing today? The weather is great, and Python is awesome. The sky is pinkish-blue. You shouldn't eat cardboard."

print(word_tokenize(EXAMPLE_TEXT))
print(sent_tokenize(EXAMPLE_TEXT))
