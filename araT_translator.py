import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch



id2label = {
    0: "مصري",
    1: "خليجي",
    2: "شامي",
    3: "شمال افريقيا"
}


TRANSLATOR_MODEL = "Zakaria279/arat5_translator_Arabic_Dialects_To_MSA"
translator_tokenizer = AutoTokenizer.from_pretrained(TRANSLATOR_MODEL)
translator_model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATOR_MODEL)




def translate(text):
    inputs = translator_tokenizer(text, return_tensors="pt")
    outputs = translator_model.generate(**inputs, max_length=100)
    return translator_tokenizer.decode(outputs[0], skip_special_tokens=True)

# ---------------------
# Streamlit App
# ---------------------
st.set_page_config(page_title="Arabic Dialect Translator", page_icon="🌍")
st.title("🌍 Arabic Dialect Translator")

st.write("أدخل نص عامي: ويترجمه للفصحى (MSA).")

user_input = st.text_area("📝 اكتب هنا:", height=120)

if st.button("ترجم 🚀"):
    if user_input.strip():
        translated_text = translate(user_input)
        st.subheader("✨ الترجمة للفصحى:")
        st.success(translated_text)
    else:
        st.warning("من فضلك أدخل نص.")
