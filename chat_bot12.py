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

    "where it is located": "MLRIT is located at Dundigal, Hyderabad, Telangana. The campus is accessible by local buses and college shuttle services.",

    "What are the admission requirements for MLRIT?": "Admission to MLRIT requires a high school diploma with at least 60% marks. Specific entrance exams are required for undergraduate and postgraduate programs.",
    "Where is MLRIT located?": "MLRIT is located at Dundigal, Hyderabad, Telangana. The campus is accessible by local buses and college shuttle services.",
    "Does MLRIT provide hostel accommodation?": "Yes, MLRIT provides hostel accommodation with various types of rooms and amenities.",
    "What is the tuition fee for B.Tech at MLRIT?": "Tuition fees for B.Tech programs are approximately INR 1,10,000 per year.",
    "What sports facilities does MLRIT offer?": "MLRIT offers basketball, football, cricket, badminton, and indoor games like table tennis and chess.",
    "Does MLRIT offer transport services?": "Yes, MLRIT provides bus services for students traveling from various parts of Hyderabad.",
    "What support services are available for students?": "MLRIT offers career counseling, academic advising, and a placement cell for student support.",
    "Are scholarships available at MLRIT?": "MLRIT provides merit-based scholarships and government scholarships for eligible students.",
    "What academic programs are offered?": "MLRIT offers undergraduate and postgraduate programs in Engineering, Technology, and Management.",
    "Does MLRIT conduct campus recruitment drives?": "Yes, MLRIT conducts campus recruitment drives every year with many top companies visiting the campus.",
    "What is the average placement package at MLRIT?": "The average placement package is INR 5 LPA, and the highest placement package is INR 15 LPA.",
    "Which companies recruit from MLRIT?": "Top companies like TCS, Infosys, Amazon, and Capgemini recruit from MLRIT.",
    "What safety policies are in place at MLRIT?": "MLRIT has 24/7 campus security, CCTV surveillance, and emergency response systems to ensure student safety.",
    "What online learning tools are available?": "MLRIT uses an LMS for online courses, recorded lectures, and assessments.",
    "What kind of food is provided in the mess at MLRIT?": "MLRIT offers a variety of vegetarian and non-vegetarian meals, including regional dishes.",
    "What is the curriculum structure for B.Tech programs?": "The B.Tech curriculum at MLRIT includes core subjects, electives, and lab-based practicals.",
    "When does the academic year start and end?": "The academic year at MLRIT typically starts in August and ends in May, with breaks in between semesters.",
    "What are the teaching facilities like at MLRIT?": "MLRIT has modern classrooms, computer labs, and specialized laboratories for each department.",
    "Are there co-curricular activities available for students?": "Yes, there are co-curricular activities like workshops, hackathons, and technical events.",
    "What clubs are available for students to join at MLRIT?": "There are various student clubs like the Coding Club, Music Club, Robotics Club, and more.",
    "Can you describe the campus of MLRIT?": "MLRIT has a 25-acre campus with academic buildings, hostels, sports facilities, and green spaces.",
    "What departments are available at MLRIT?": "MLRIT offers departments in Engineering, Technology, Management, and Humanities, among others.",
    "What is student life like at MLRIT?": "Student life at MLRIT is vibrant with numerous clubs, events, and activities.",
    "How can I apply for internships?": "Students can apply for internships through the Placement Cell or by contacting companies directly.",
    "Does MLRIT offer research opportunities?": "Yes, MLRIT has research opportunities in various engineering and technology fields.",
    "How qualified are the faculty members?": "MLRIT has highly qualified faculty, many with Ph.D. degrees in their respective fields.",
    "What extracurricular activities are available?": "There are various extracurricular activities like music, dance, drama, and sports.",
    "Does MLRIT collaborate with industries?": "MLRIT collaborates with industry partners for research, internships, and placements.",
    "Does MLRIT accept international students?": "Yes, MLRIT welcomes international students and provides necessary support.",
    "How can I connect with MLRIT alumni?": "MLRIT has an active alumni network for professional and social connections.",
    "Does MLRIT provide counseling services?": "Yes, MLRIT provides counseling services for academic and mental health support.",
    "What are the library services at MLRIT?": "MLRIT's library offers digital resources, books, journals, and a peaceful study environment.",
    "What career guidance services are available?": "MLRIT provides career guidance through workshops, seminars, and one-on-one counseling.",
    "Does MLRIT provide mental health support?": "Yes, MLRIT has mental health support and counseling services for students.",
    "What green initiatives are at MLRIT?": "MLRIT focuses on sustainability with green initiatives like solar panels and rainwater harvesting.",
    "What events and festivals are held at MLRIT?": "MLRIT hosts annual cultural fests, technical events, and sports meets for students.",
    "Are there any student exchange programs?": "Yes, MLRIT has student exchange programs with partner universities.",
    "Is MLRIT accredited?": "MLRIT is accredited by NBA and NAAC, ensuring quality education.",
    "Are there part-time jobs available?": "MLRIT offers part-time job opportunities through campus events and student-run initiatives.",
    "Does MLRIT have visiting faculty?": "MLRIT has visiting faculty members with industry experience who share their expertise.",
    "What are the dining options in the hostel?": "MLRIT's hostel dining services offer nutritious meals with a variety of options.",
    "What can I do on weekends at MLRIT?": "On weekends, students can participate in cultural fests, workshops, and sports activities.",
    "Does MLRIT have any publications or journals?": "MLRIT has a student magazine, and students contribute to publications and research journals.",
    "What student clubs are at MLRIT?": "There are multiple student clubs focused on technology, literature, music, and sports.",
    "What sustainability initiatives does MLRIT have?": "MLRIT is committed to sustainability with eco-friendly campus practices.",
    "What is the IT infrastructure like at MLRIT?": "MLRIT provides state-of-the-art IT infrastructure for students and staff.",
    "What labs does MLRIT have?": "The college has modern laboratories, including those for research and development.",
    "Are there public speaking and debate opportunities?": "There are opportunities for public speaking and debate in clubs and competitions.",
    "What is the current ranking of MLRIT?": "MLRIT is ranked among the top engineering colleges in India for its quality education.",
    "When was MLRIT founded?": "MLRIT was established in 2005 and has grown rapidly as a prestigious institution."

    


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
