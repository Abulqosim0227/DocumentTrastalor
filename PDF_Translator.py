import streamlit as st
from pdfminer.high_level import extract_text
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from langchain.prompts import PromptTemplate
from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
OPENAI_API_KEY = "sk-proj-4L7plzjsl96L3r1BwmkXT3BlbkFJkpxZXh6lomH"
model_name='gpt-3.5-turbo'
llm = ChatOpenAI(model_name=model_name,temperature=0,openai_api_key=OPENAI_API_KEY)

languages = {
    'Afrikaans': 'af', 'Albanian': 'sq', 'Amharic': 'am', 'Arabic': 'ar', 'Armenian': 'hy', 'Azerbaijani': 'az', 
    'Basque': 'eu', 'Belarusian': 'be', 'Bengali': 'bn', 'Bosnian': 'bs', 'Bulgarian': 'bg', 'Catalan': 'ca', 
    'Cebuano': 'ceb', 'Chichewa': 'ny', 'Chinese (Simplified)': 'zh-CN', 'Chinese (Traditional)': 'zh-TW', 
    'Corsican': 'co', 'Croatian': 'hr', 'Czech': 'cs', 'Danish': 'da', 'Dutch': 'nl', 'English': 'en', 'Esperanto': 'eo', 
    'Estonian': 'et', 'Filipino': 'tl', 'Finnish': 'fi', 'French': 'fr', 'Frisian': 'fy', 'Galician': 'gl', 
    'Georgian': 'ka', 'German': 'de', 'Greek': 'el', 'Gujarati': 'gu', 'Haitian Creole': 'ht', 'Hausa': 'ha', 
    'Hawaiian': 'haw', 'Hebrew': 'he', 'Hindi': 'hi', 'Hmong': 'hmn', 'Hungarian': 'hu', 'Icelandic': 'is', 
    'Igbo': 'ig', 'Indonesian': 'id', 'Irish': 'ga', 'Italian': 'it', 'Japanese': 'ja', 'Javanese': 'jv', 'Kannada': 'kn', 
    'Kazakh': 'kk', 'Khmer': 'km', 'Korean': 'ko', 'Kurdish (Kurmanji)': 'ku', 'Kyrgyz': 'ky', 'Lao': 'lo', 
    'Latin': 'la', 'Latvian': 'lv', 'Lithuanian': 'lt', 'Luxembourgish': 'lb', 'Macedonian': 'mk', 'Malagasy': 'mg', 
    'Malay': 'ms', 'Malayalam': 'ml', 'Maltese': 'mt', 'Maori': 'mi', 'Marathi': 'mr', 'Mongolian': 'mn', 'Burmese': 'my', 
    'Nepali': 'ne', 'Norwegian': 'no', 'Pashto': 'ps', 'Persian': 'fa', 'Polish': 'pl', 'Portuguese': 'pt', 'Punjabi': 'pa', 
    'Romanian': 'ro', 'Russian': 'ru', 'Samoan': 'sm', 'Scots Gaelic': 'gd', 'Serbian': 'sr', 'Sesotho': 'st', 
    'Shona': 'sn', 'Sindhi': 'sd', 'Sinhala': 'si', 'Slovak': 'sk', 'Slovenian': 'sl', 'Somali': 'so', 'Spanish': 'es', 
    'Sundanese': 'su', 'Swahili': 'sw', 'Swedish': 'sv', 'Tajik': 'tg', 'Tamil': 'ta', 'Telugu': 'te', 'Thai': 'th', 
    'Turkish': 'tr', 'Ukrainian': 'uk', 'Urdu': 'ur', 'Uzbek': 'uz', 'Vietnamese': 'vi', 'Welsh': 'cy', 'Xhosa': 'xh', 
    'Yiddish': 'yi', 'Yoruba': 'yo', 'Zulu': 'zu'
}

# Function to add page numbers to PDF
def add_page_number(canvas, doc):
    page_num = canvas.getPageNumber()
    text = "Page %s" % page_num
    canvas.drawRightString(200*mm, 20*mm, text)

# Function to preprocess text
def preprocess(text_chunk,language):
    prompt_templateFAQS = """
   I have to dump given text in a pdf so add new lines and end of paragraphs \
    accordingly. Also add a new line at the end of your response. \
    Translate the following text to {target_language} :\n{text}
    """
    PROMPTFAQS = PromptTemplate(
        template=prompt_templateFAQS, input_variables=["target_language", "text"]
    )
    chain = LLMChain(llm=llm , prompt=PROMPTFAQS)
    output = chain.predict(target_language=language, text=text_chunk)
    return output

# Used for margins
inch = 72

# File uploader allows user to upload their PDF
uploaded_file = st.file_uploader("Upload a PDF", type=['pdf'])
selected_language = st.selectbox('Select a language:', list(languages.keys()))
# Text field allows user to add a line to end of each chunk

# Check if a PDF has been uploaded
if uploaded_file is not None:
    # Create a button to process the PDF
    process_button = st.button("Process PDF")

    if process_button:
        with st.spinner('Processing...'):
            # Use PDFMiner to extract text from PDF
            print(uploaded_file.name)
            text = extract_text(uploaded_file.name)

            # Split text into chunks of 500 characters
            # text_chunks = [text[i: i + 500] for i in range(0, len(text), 500)]
            # Initialize progress bar and text area after the button to ensure they remain static
            text_chunks = [text[i:i+1600] for i in range(0, len(text), 1600)]
            progress_bar = st.empty()
            text_area = st.empty()
            
            # Lowercase and delay each chunk
            for i, text_chunk in enumerate(text_chunks):
                preprocessed_chunk = preprocess(text_chunk,selected_language)
                text_area.text_area("Translated Text", preprocessed_chunk, height=200, key=str(i))


                # Update progress bar
                progress = (i + 1) / len(text_chunks)
                progress_bar.progress(progress)
else:
    st.write("Please upload a PDF.")