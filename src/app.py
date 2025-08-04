import streamlit as st
import joblib
import os

# تحميل النموذج و TF-IDF vectorizer
@st.cache_resource
def load_model_and_vectorizer(model_path="models/sentiment_model.pkl", vectorizer_path="models/tfidf_vectorizer.pkl"):
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        st.error("❌ تأكد من وجود ملفات النموذج والـ vectorizer داخل مجلد models/")
        return None, None
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer


# استخراج الكلمات الإيجابية والسلبية
POSITIVE_WORDS = {"good", "great", "excellent", "love", "wonderful", "best", "happy"}
NEGATIVE_WORDS = {"bad", "terrible", "worst", "hate", "awful", "sad", "poor"}

def highlight_words(text):
    tokens = text.lower().split()
    pos = [w for w in tokens if w in POSITIVE_WORDS]
    neg = [w for w in tokens if w in NEGATIVE_WORDS]
    return pos, neg


# واجهة Streamlit
def main():
    st.set_page_config(page_title="Sentiment Analyzer", layout="centered")
    st.title("🧠 Amazon Product Review Sentiment Analyzer")

    st.markdown("""
    أدخل مراجعة منتج وسيقوم النموذج بتحليل المشاعر لتحديد ما إذا كانت **إيجابية** أو **سلبية**.
    """)

    model, vectorizer = load_model_and_vectorizer()
    user_input = st.text_area("📝 اكتب المراجعة هنا:", height=150)

    if st.button("🔍 تحليل المشاعر") and model and vectorizer:
        if not user_input.strip():
            st.warning("⚠️ الرجاء إدخال مراجعة.")
            return

        with st.spinner("⏳ جاري التحليل..."):
            try:
                input_vector = vectorizer.transform([user_input])
                prediction = model.predict(input_vector)[0]
                prob = model.predict_proba(input_vector)[0]
            except Exception as e:
                st.error(f"❌ حصل خطأ أثناء التنبؤ: {e}")
                return

        label = "إيجابي ✅" if prediction == 1 else "سلبي ❌"
        st.metric("🔎 النتيجة:", label)

        st.subheader("📈 نسبة الثقة:")
        st.progress(float(prob[1]))
        st.write(f"🟢 احتمال إيجابي: {prob[1]:.2%}")
        st.write(f"🔴 احتمال سلبي: {prob[0]:.2%}")

        pos_words, neg_words = highlight_words(user_input)
        st.subheader("📌 الكلمات المميزة في النص:")
        st.write(f"**كلمات إيجابية:** {', '.join(pos_words) if pos_words else 'لا توجد'}")
        st.write(f"**كلمات سلبية:** {', '.join(neg_words) if neg_words else 'لا توجد'}")

    st.sidebar.markdown("## ⚙️ إعدادات")
    if st.sidebar.button("🔁 إعادة تحميل النموذج"):
        st.cache_resource.clear()
        st.experimental_rerun()


if __name__ == "__main__":
    main()
