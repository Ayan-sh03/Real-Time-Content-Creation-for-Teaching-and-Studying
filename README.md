# Real-time Content Creation System for Lecturers

## Overview

The need for cutting-edge tools in the educational landscape to improve teaching and learning is constantly growing. Conventional techniques for creating lecture content frequently rely on human input, which reduces the effectiveness and dynamic nature of instruction. Our initiative intends to address this difficulty by providing lecturers with a real-time content production system, thereby revolutionizing the art of lecture creation. By utilizing state-of-the-art technologies like text summarization, real-time picture search, and speech recognition, this system provides a smooth and entertaining method of creating content. Through the integration of these characteristics, we aim to overcome the constraints of conventional approaches and provide lecturers with immediate access to a variety of visual aids, thus enabling them to effortlessly present engaging and enlightening lectures. Our goal with this initiative is to improve students' educational experiences.

## Key Features

1. **Speech Recognition**
   - Implement advanced speech recognition technology to accurately capture lecturers' speech in real-time. This feature ensures seamless conversion of spoken content into text, reducing the need for manual input.

2. **Real-time Image Search**
   - Integrate with online image databases to instantly retrieve relevant images based on the content of the lecture. This feature enhances visual aids and engages students with diverse and captivating visuals.

3. **Interactive Visualization**
   - Enable interactive visualization tools to enhance understanding and engagement. This may include features such as zooming, highlighting, and annotation functionalities for both lecturers and students.

4. **Customization Options**
   - Provide customization options for lecturers to tailor the content creation process according to their preferences. This may include adjusting the layout, font styles, color schemes, and other visual elements of the generated content.

5. **Multi-platform Support**
   - Ensure compatibility across various devices and platforms, including desktop computers, laptops, tablets, and smartphones. This enables flexibility in accessing and delivering content, catering to diverse learning environments and preferences.

## Functionalities

1. **Registration and Login**
   - In order to use the content creation system, lecturers need to register on the website.

2. **Session Initialization**
   - After logging in, instructors start a new session specifically for the lesson they plan to create.
   - They input the required information, including the lecture title, subject, and time and date.

3. **Speech Recognition**
   - After starting the session, lecturers activate the speech recognition feature. 
   - The system listens to the lecturer's voice input during the lecture and converts it into text data in real-time.

4. **Real-time Image Extraction**
   - The system pulls relevant photos for the lecture as the speaker speaks by using the Google API.
   - Only the best pictures are shown once the retrieved images are sorted according to quality and relevancy.

5. **Lecture Playground Area**
   - A designated area within the content creation interface serves as the "lecture playground."
   - Here, the relevant images extracted by the system are dynamically displayed in real-time alongside the transcribed text of the lecture.

6. **Pause and Explanation**
   - Instructors can choose to stop the speech recognition function at any point when creating a lecture.
   - If an instructor wants to go into further depth about a recovered image, they can interrupt voice recognition and give further background or explanation.

7. **Video Summarization**
   - Upon completion of the lecture, a summarized video of the lecture is generated.
   - The system compiles the transcribed text, images retrieved, and any additional multimedia elements into a concise video format.

## Flowchart

Below is a flowchart that illustrates the process flow of the real-time content creation system:

![Flowchart](https://github.com/user-attachments/assets/f06d95a9-0526-4e3f-bca8-f46b4d07a86e)

##

## User Interface

![image](https://github.com/user-attachments/assets/84ffcf49-0470-4b60-9053-7d09dd12b6be)

 illustrates the central Content Creation User Interface (UI), where educational videos are showcased alongside a dedicated subscript section. This UI serves as the primary workspace for real-time content creation, providing users with dynamic video playback controls and an integrated subscript feature for enhancing comprehension. Through seamless synchronization between the video display area and the subscript section, users can engage with educational material efficiently while accessing supplementary information and annotations.The UI's customizable options and potential collaborative features foster an interactive and immersive learning experience, empowering educators and learners to create and engage with educational content effectively. 

## 
## TECHNOLOGY USED
### Front End Development

**Next.js**
- Streamlines UI development for an interactive, smooth, and user-friendly interface.
- Server-side Rendering: Pre-renders pages on the server for faster initial load times and better SEO.
- Static Site Generation (SSG): Pre-generates HTML at build time for fast page loads and reduced server load.
- Automatic Code Splitting: Efficiently loads only the necessary code for each page, enhancing performance.

**CSS Frameworks**
- Modular CSS & Tailwind CSS: Provides responsive and customizable user interfaces with pre-defined utility classes for consistent branding and design.

### Backend Development

**Frameworks**
- Django: Primary framework for building robust and scalable web applications.
- Flask: Used for rapid prototyping and flexibility.
**Database**
- PostgreSQL: Reliable and scalable database management system for storing and retrieving data efficiently.
**Real-time Communication**
- Socket.io: Enables real-time communication between the backend and frontend, enhancing user engagement.
**Natural Language Processing**
- Gensim & Transformers: Used for Efficient Key word extraction
  
### External APIs
**Google's Dynamic Image Search API**
- Ensures real-time fetching of images, keeping search capabilities up-to-date.

## SYSTEM OVERVIEW
### 1.Modules

**1.2 Socket Setup and Events**
- Establishes WebSocket connections between server and clients.
- Handles 'connect', 'message', and 'disconnect' events for real-time image streaming.

**1.3 Keyword Extraction and Embedding**
- Preprocesses text using Python’s string module.
- Uses Gensim’s Word2Vec for semantic keyword extraction, enhancing contextual relevance.

**1.4 Image Fetching and Preprocessing**
- Fetches images using Google Custom Search API based on keywords.
- Processes images to a default size of 224x224 pixels, converting them to NumPy arrays.

**1.5 Analysis with CLIP Module**
- Uses CLIP model to rank images by semantic relevance to the given keyword.
- Ensures selected images are contextually appropriate and visually appealing.

**1.6 Result**
- Outputs a list of relevant images, ranked by relevance to keywords, ready for frontend display.

### 2.Integration

**2.1 WebSockets**
- Enables real-time image streaming from the backend to the frontend.
- Provides a dynamic and interactive user experience with instant updates.

**2.2 REST API**
- Facilitates communication between client and server for image fetching and CRUD operations.
- Ensures efficient and scalable interaction for various application functionalities.

## References

1.	P. M. Ashok Kumar, T. Subha Mastan Rao, L. Arun Raj, and E. Pugazhendi, 
“An Efficient Text-Based Image Retrieval Using Natural Language Processing(NLP)              Techniques”. Intelligent System Design (pp . 505-519) January 2021. 
     https://www.researchgate.net/publication/343585352

2.	Rohit Ranchal, Member, Teresa Taber-Doughty, Yiren Guo, Keith Bain, Heather Martin, J. Paul Robinson, Member, and Bradley S. Duerstock, “Using Speech Recognition for Real-Time Captioning and Lecture Transcription in the Classroom.” IEEE TRANSACTIONS ON LEARNING TECHNOLOGIES,VOL.6,NO. 4, OCTOBER-DECEMBER 2013.
      	https://ieeexplore.ieee.org/document/6529071

3.	Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever, “Learning Transferable Visual Models From Natural Language Supervision.” International Conference on Machine Learning, 2021
      	https://arxiv.org/pdf/2103.00020 
                      
4.	Anmol Bhat, Aneesh C.Rao, Anirudh Bhaskar, V. Adithya, D. Pratiba,
“A  Cost-Effective Audio-Visual Summarizer for Summarization of Presentations and 
Seminars.” 3rd IEEE International Conference on Computational Systems and Information Technology for Sustainable Solutions 2018.
      	https://ieeexplore.ieee.org/document/8768740      

