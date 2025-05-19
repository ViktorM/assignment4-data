from cs336_data.utils import identify_language

text = "This is a sample text in English."
lang_code, confidence = identify_language(text, "../data/lid.176.bin")

print(f"Detected language: {lang_code}, confidence: {confidence:.3f}")

text = "Hola, ¿cómo estás?"
print(identify_language(text, "../data/lid.176.bin" ))

text = "年齡: 28 歲 區域: 高雄市"
print(identify_language(text, "../data/lid.176.bin"))
