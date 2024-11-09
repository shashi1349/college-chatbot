import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer, util
import pickle

college_data = {
    "Hi":"Hello! how can i assist you",
    "hello":"hi, how can i help you today",
    "About MLRIT": "Maturi Laxmi Reddy Institute of Technology (MLRIT) is a reputed engineering college located in Hyderabad, India. "
                    "It offers undergraduate and postgraduate courses in various branches of engineering and is affiliated with Osmania University.",
    
    "Courses Offered": "MLRIT offers B.Tech programs in Computer Science, Electronics, Electrical, Civil, Mechanical, and Information Technology. "
                       "It also offers M.Tech programs in Computer Science, VLSI, Structural Engineering, and Power Systems.",
    
    "Admissions": "Admission to MLRIT is based on the TS EAMCET for undergraduate programs and TS PGECET for postgraduate programs. "
                  "The college also accepts GATE scores for M.Tech admissions. Candidates must apply through the official website.",
    
    "Campus Facilities": "MLRIT provides state-of-the-art facilities such as Wi-Fi, a well-stocked library, hostels for boys and girls, sports complexes, "
                          "and advanced laboratories in all departments. The campus also features an auditorium, seminar halls, and a cafeteria.",
    
    "Placements": "MLRIT has an excellent placement record, with top companies such as TCS, Infosys, Accenture, Cognizant, Wipro, and Amazon regularly visiting "
                  "the campus for recruitment. The average salary package is competitive, and the placement cell organizes training and mock interviews."
                  "The average placement package is INR 5 LPA, and the highest placement package is INR 15 LPA.",
    
    "Contact Information": "For any queries, you can contact MLRIT via email at info@mlrit.ac.in or visit the official website at www.mlrit.ac.in.",
    
    "Faculty": "MLRIT has a highly qualified and experienced faculty, with many professors holding PhDs in their respective fields. The faculty members are "
               "actively involved in research and development, and the college frequently hosts seminars, workshops, and conferences.",
    
    "Student Clubs": "MLRIT has a vibrant student life, with several student clubs and societies, including technical clubs (like the Robotics Club), cultural clubs, "
                     "sports clubs, and a literary society. Students are encouraged to participate in extracurricular activities, and the college organizes annual festivals.",
    
    "Annual Events": "MLRIT hosts a variety of events throughout the year, including TECHNIX (a technical fest), SPARX (the annual cultural fest), and sports tournaments. "
                     "These events see active participation from students across the college, as well as from other institutions.",
    
    "Hostel Facilities": "MLRIT provides separate hostel facilities for boys and girls. The hostels are equipped with modern amenities such as Wi-Fi, a common dining hall, "
                         "24/7 water and electricity supply, and recreational areas. Rooms are available on both single and sharing basis.",
    
    "Library": "The MLRIT library is a spacious, multi-storey building with a wide collection of books, journals, e-books, and research papers. "
               "It also provides digital resources and access to online databases like IEEE, Springer, and Elsevier.",
    
    "When was MLRIT founded?": "MLRIT was established in 2005 and has grown rapidly as a prestigious institution.",

    "Tuition fee ": "Tuition fees for B.Tech programs are approximately INR 1,10,000 per year.",
    
    "Transport services": "Yes, MLRIT provides bus services for students traveling from various parts of Hyderabad.",

    "where college is located": "MLRIT is located at Dundigal, Hyderabad, Telangana. The campus is accessible by local buses and college shuttle services."
    
    


}


# Initialize the model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Encode the questions
questions = list(college_data.keys())
question_embeddings = model.encode(questions, convert_to_tensor=True)

# Function to get the most relevant answer using cosine similarity with a threshold
def get_answer(query, threshold=0.7):
    query_embedding = model.encode(query, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(query_embedding, question_embeddings)[0]
    cosine_scores = cosine_scores.cpu().numpy()
    top_match_idx = np.argmax(cosine_scores)
    max_score = cosine_scores[top_match_idx]

    if max_score >= threshold:
        top_match_question = questions[top_match_idx]
        return college_data[top_match_question]
    else:
        return "I'm sorry, I couldn't find an answer to your question. Could you try rephrasing it?"

# Initialize chat history in session stater 
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Streamlit Sidebar for App Instructions
st.sidebar.title("MLRIT College Chatbot")
st.sidebar.write("""
This chatbot can answer questions related to MLRIT College, such as admissions, fees, facilities, and more.
- **Type your question** in the input box below.
- **Press Enter** to get a response.
- Use the **Clear Chat History** button to start fresh.
""")
st.sidebar.write("**Tip**: Try asking specific questions like 'What is the campus location?' or 'Does MLRIT offer scholarships?'")

# Streamlit UI with header and clear chat history button
st.title("📚 MLRIT College Chatbot")
st.write("Welcome! Ask me anything about MLRIT College.")
if st.button("Clear Chat History"):
    st.session_state['chat_history'] = []

# Chat input from user
user_input = st.text_input("Your Question:")

# Generate response if there is user input
if user_input:
    answer = get_answer(user_input)
    st.session_state['chat_history'].append({"question": user_input, "answer": answer})

# Display chat history with improved UI
st.markdown("### Chat History")
for chat in st.session_state['chat_history']:
    st.write(f"**🧑‍🚒 You:** {chat['question']}")
    st.markdown(f"<div style='padding: 10px; background-color: black; border-radius: 10px;'>"
                f"<b>🤖 MLRIT Chatbot:</b> {chat['answer']}</div>", unsafe_allow_html=True)
    st.write("---")
