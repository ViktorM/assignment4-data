from cs336_data.utils import identify_language, classify_nsfw, classify_toxic_speech

text = "This is a sample text in English."
lang_code, confidence = identify_language(text, "../data/lid.176.bin")

print(f"Detected language: {lang_code}, confidence: {confidence:.3f}")

text = "Hola, ¿cómo estás?"
print(identify_language(text, "../data/lid.176.bin" ))

text = "年齡: 28 歲 區域: 高雄市"
print(identify_language(text, "../data/lid.176.bin"))

print(classify_nsfw("Hot group anal sex with 3 girls.", "../data/jigsaw_fasttext_bigrams_nsfw_final.bin"))
print(classify_toxic_speech("You are horrible, kill yourself!", "../data/jigsaw_fasttext_bigrams_hatespeech_final.bin"))
