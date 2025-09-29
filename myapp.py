import streamlit as st
import torch
import tempfile
from transformers import pipeline, WhisperProcessor, AutoModelForCausalLM, AutoTokenizer


st.set_page_config(page_title="Whisper + Qwen",page_icon='🤖')
st.title("AI Voice Assistant: Transcribe, Translate & Chat")
st.markdown("***Add or record audio to transcribe or translate your message***")


if "messages" not in st.session_state:
    st.session_state['messages']=[]

if "show_upload" not in st.session_state:
    st.session_state['show_upload']=[]

if "show_record" not in st.session_state:
    st.session_state['show_record']=[]

@st.cache_resource(show_spinner=False)
def load_asr(model_name="openai/whisper-small"):
    return pipeline("automatic-speech-recognition",
                    model=model_name,
                    device=-1,
                    torch_dtype=torch.float32)
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
asr=load_asr()

@st.cache_resource(show_spinner=False)
def load_qwen(model_name="Qwen/Qwen3-0.6B"):
    tokenizer=AutoTokenizer.from_pretrained(model_name)
    model=AutoModelForCausalLM.from_pretrained(model_name,
                                               torch_dtype='auto',
                                               device_map='auto')
    return tokenizer,model
tokenizer,model=load_qwen()

def run_qwen(user_text,tokenizer,model,prompt="copy the text exactly as is, with no extra modifications or changes to the original text."):
    messages=[
        {"role":"system","content":prompt},
        {"role":"user","content":user_text}
    ] 

    prompt=tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False)
    model_inputs=tokenizer([prompt],return_tensors='pt').to(model.device)
    output=model.generate(
        **model_inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7, 
        top_p=0.95
    )
    generated=output[0][model_inputs['input_ids'].shape[-1]:]
    content=tokenizer.decode(generated, skip_special_tokens=True).strip()
    return content


@st.cache_data(show_spinner=False)
def transcribe_translate(path):
    transcription=asr(path)
    translation=asr(path, generate_kwargs={'task':'translate'})
    return translation['text'],transcription['text']


cont=st.container()
with cont:
    tab1,tab2=st.tabs(["Upload","Record"])
    with tab1:
        st.markdown("**Upload Audio**")
        audio_file= st.file_uploader("Upload",label_visibility="collapsed")
    with tab2:
        st.markdown("**Record Audio**")
        vm= st.audio_input("Record",label_visibility="collapsed")


    audio_bytes = None
    if vm is not None:
        audio_bytes = vm.getvalue()
    elif audio_file is not None:
        audio_bytes = audio_file.getvalue()

    if audio_bytes:
        st.audio(audio_bytes)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            path=tmp.name
        
        with st.spinner("Running Whisper..."):
            translation,transcription=transcribe_translate(audio_bytes)
        st.session_state['transcription_text']=transcription
        st.session_state['translation_text']=translation

        # st.markdown("Here is your audio transcribed:")
        # st.code(transcription,language=None)
        # st.markdown("Here is your audio translated (English):")
        # st.code(translation,language=None)


    options=st.selectbox(
        "Choose a Task", 
        ("Transcribe","Translate"),
        index=None,
        placeholder="Select task")
    run_task=st.button("Run Task")


    if run_task and options: 
        if options=='Transcribe':
            text=st.session_state.get('transcription_text')
            if text:
                reply=run_qwen(text,tokenizer,model)
                with st.chat_message("assistant"):
                    st.markdown("Transcribed Text:")
                    st.write(reply)
            else:
                st.warning("No transcription available. Please upload an audio")


        if options=='Translate':
            text=st.session_state.get('translation_text')
            if text:
                reply=run_qwen(text,tokenizer,model)
                with st.chat_message("assistant"):
                    st.markdown("Translated Text:")
                    st.write(reply)
            else:
              st.warning("No translation available. Please upload an audio.")

    st.divider()          
    col_left,col_right=st.columns([10,1],vertical_alignment="bottom")
    with col_left:
        st.markdown("***Or chat with Qwen***")
    with col_right:
        if st.button("📝",help="New Chat"):
            st.session_state['messages']=[]
    scroll_chat=st.container(height=500,border=True)
    with scroll_chat:
        for msg in st.session_state.messages:
            with st.chat_message(msg['role']):
              st.markdown(msg['content'])
    typed=st.chat_input("Type your message")
    if typed:
        st.session_state.messages.append({'role':'user','content':typed})
        with scroll_chat:
            with st.chat_message("user"):
                st.markdown(typed)

           
        reply=run_qwen(typed,tokenizer,model,prompt="be a friendly and conversing assistant")
        st.session_state.messages.append({"role":'assistant','content':reply})
        with scroll_chat:
            with st.chat_message("assistant"):
                st.markdown(reply)