import streamlit as st
from engine import inference

def ui():
    st.markdown("# Image Captioning")
    st.markdown("***A sequence to sequence model to caption images built using PyTorch.\
    This model uses inceptionV3 as encoder and LSTM layers as decoder.\
    This model is trained on Flickr30k dataset.***")

    st.markdown("# Examples")
    st.image('./test_examples/surfing.png', width = 500, channels = 'RGB')
    st.markdown('**PREDICTION:** ***a man in a wetsuit surfing .***')
    st.markdown('')
    st.image('./test_examples/dirt_bike.png', width = 500, channels = 'RGB')
    st.markdown('**PREDICTION:** ***a man in a blue helmet is riding a dirt bike on a dirt track .***')
    st.markdown('')
    st.image('./test_examples/dog.png', width = 500, channels = 'RGB')
    st.markdown('**PREDICTION:** ***a dog is running on the beach .***')
    st.markdown('')
    st.markdown('# Try it out:')
    uploaded_file = st.file_uploader("Upload an Image", type=['png', 'jpeg', 'jpg'])

    if uploaded_file is not None:
        out = inference(uploaded_file)
        st.image(uploaded_file, width = 500, channels = 'RGB')
        st.markdown("**PREDICTION:** " + out)

    st.markdown('## [Code on GitHub](https://github.com/Koushik0901/Image-Captioning)')
    st.markdown('')
    st.markdown("""# Connect with me
  [<img height="30" src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" />][github]
  [<img height="30" src="https://img.shields.io/badge/linkedin-blue.svg?&style=for-the-badge&logo=linkedin&logoColor=white" />][LinkedIn]
  [<img height="30" src = "https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white"/>][instagram]
  
  [github]: https://github.com/Koushik0901
  [instagram]: https://www.instagram.com/koushik_shiv/
  [linkedin]: https://www.linkedin.com/in/koushik-sivarama-krishnan/""", unsafe_allow_html=True)
if __name__ == '__main__':
    ui()
