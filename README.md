\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{enumerate}
\usepackage{graphicx}
\usepackage[utf8]{inputenc}
\usepackage{subcaption}
\usepackage{hyperref}
\usepackage{float}
\usepackage[a4paper, total={6in, 10in}]{geometry}
\usepackage{tabularx}
\usepackage[table]{xcolor}
\newcolumntype{P}[1]{>{\arraybackslash}p{#1}}

\usepackage[style=numeric, backend=biber,  sorting=none]{biblatex}
\addbibresource{references.bib}

\title{Designing a secure messaging application with biometric identification computing on homomorphic encrypted data.}
\author{Alexander Stæhr Johansen, 201905865@post.au.dk \\
Henrik Tambo Buhl, 201905590@post.au.dk}
\date{16/02/2022}
\begin{document}
\maketitle
\section*{Introduction}
The purpose of the following document is to introduce the bachelor's project "Designing a secure messaging client-server application with biometric identification computing on homomorphic encrypted data".

\section*{Motivation}
In the present socio-economic environment, people are increasingly dependent on electronic means of communication and verification. This has led public and private institutions to adopt specialized internal communication channels, accessed via. biometric identification, as it is more secure and accurate than traditional forms of distribution of information and methods for authentication.

Therefore, it is of significant interest to build a secure communication system which can verify a person, without compromising the privacy of the individual.

\section*{Objectives}
The main tasks are to create a secure proof-of-concept messaging web application, with security ensured via the Signal protocol and its underlying XEdDSA and VXEdDSA signatures, Double Ratchet algorithm, X3DH key agreement protocol and Sesame algorithm \cite{signalproto}, and a biometric identification system which uses facial recognition, implemented via Eigenfaces \cite{7340029}, as biometric parameter for verification, and homomorphic encryption \cite{10.1007/978-3-030-78086-9_4}, implemented via Microsoft SEAL \cite{sealcrypto}, for privacy-preserving computation. Sub-tasks include: 
\begin{itemize}
    \item Document the proof-of-concept with a software requirements specification, an architectural document, and a test specification.
    \item Verify and document the testing of the software requirements specification.
    \item Discuss what improvements could be made with respect to usability, accuracy and scalability.
\end{itemize}

\section*{Work plan}
To accommodate the above objectives, the following work plan 
will be executed. The work plan is tentative and as such changes will occur. The person responsible for each deliverable has his initial suffixed after this. For example, deliverables with the suffix "(A)" must be created by Alexander Stæhr Johansen. \\
The Django tutorials referenced below are to be found \href{https://channels.readthedocs.io/en/latest/tutorial/}{\underline{here}} \cite{Django_tutorials}. The CKKS introduction can be found \href{https://blog.openmined.org/ckks-explained-part-1-simple-encoding-and-decoding/}{\underline{here}} \cite{CKKS_introduction}, and the Python binding used for implementing the Microsoft SEAL library is found \href{https://github.com/Huelse/SEAL-Python}{\underline{here}} \cite{SEAL-Python}. 

\begin{flushleft}
\begin{table}[H]
\centering
\begin{tabular}{|P{4.5cm}|P{4.5cm}|P{5cm}|}
\hline
\centering \textbf{\LARGE{Week}} & \centering\textbf{\LARGE{Agenda}}   & \ \ \ \ \ \   \textbf{\LARGE{Deliverables}} \\ \hline %quick fix
\textbf{W5:} Introduction. & Do project description refinement.   & \textbf{D1:} Refined project description. (A and H)    \\ \hline
\rowcolor{lightgray}
\textbf{W6:} Technological introduction. & Complete Django tutorial 1 and 2. Setup environment for SEAL-Python.   & \textbf{D2:} Basic chat application. (A)    \\ \hline
\textbf{W7:} Technological introduction, thesis writing, specification and Eigenface implementation. 
& Complete Django tutorial 3, execute the first iteration of the requirement specification, implement PCA and write the sections: "PCA", "power method" and "Eigen shift procedure". & \textbf{D3:} Asynchronous chat application with two-factor verification. (A)
\textbf{D4:} 1st draft of the requirements specification. (A) 
\textbf{D5:} PCA code. (H)
\textbf{D6:} PCA section. (H)
\textbf{D7:} Power method section. (H) 
\textbf{D8:} Eigen shift procedure. (H) 
\\ \hline
\textbf{W8:} Technological introduction, thesis writing, specification and Eigenface implementation. & \raggedright Complete Django tutorial 4, execute the first iteration of the architectural specification, implement Goldschmidt's algorithm, Eigenface facial recognition and write the corresponding sections: "Goldschmidt's algorithm" and "Eigenface facial recognition. & \textbf{D9:} Automated test suite for chat application. (A)
\textbf{D10:} 1st draft of the architectural specification. (A)
\textbf{D11:} Goldschmidt's algorithm code. (H)
\textbf{D12:} Goldschmidt's algorithm section. (H)
\textbf{D13:} Facial recognition code. (H)
\textbf{D14:} Facial recognition section. (H)
 \\ \hline
 
 \textbf{W9:} Thesis writing and Eigenface implementation. & Write the sections: “purpose”, “problem definition”, "related work" and “software development processes” of the thesis. Implement vector operations and write the corresponding section. & \textbf{D15:} Purpose section. (A) \textbf{D16:} Problem definition section. (A) \textbf{D17:} Related work section. (A and H)
 \textbf{D18:} Software development processes section. (A) \textbf{D19:} Vector operations code. (H) \textbf{D20:} Final Eigenface implementation. (H) \\ \hline
 \rowcolor{lightgray}
 
 \textbf{W10:} Specification and homomorphic encryption. & Execute the first iteration of the test specification and begin the introduction to CKKS in SEAL-Python. & \textbf{D21:} First draft of the test specification. (A) \\ \hline
 

\textbf{W11:} Thesis writing, Signal protocol implementation and homomorphic encryption. & \raggedright Write the section: “Signal protocol” and its associated subsections: "WEdDSA and VXEdDSA", "X3DH", "Double Ratchet" and "Sesame" and begin the implementation of the protocol. Introduction to CKKS in SEAL-Python continued. & \textbf{D22:} WEdDSA and VXEdDSA. (A)
\textbf{D23:} X3DH. (A)
\textbf{D24:} Double Ratchet. (A)
\textbf{D25:} Sesame. (A)
\\ \hline

\textbf{W12:} Signal protocol implementation continued. Homomorphic Eigenfaces implementation. & Implement Signal protocol. Start with implementing R2. & \textbf{D26:} Signal protocol implemented. (A) \\ \hline

\raggedright \textbf{W13:} Write requirements specification and continue to implement homomorphic Eigenfaces. & Execute the second iteration of the requirements specification. Start to implement homomorphic eigen shift procedure. & \textbf{D27:} Second draft of the requirements specification. (A) \\ \hline

%\multicolumn{3}{c}{Break / catch-up.} \\ \hline

%
\end{tabular}
%\caption{\label{}}
\end{table}
\end{flushleft}
\begin{flushleft}
\begin{table}[H]
\centering
\begin{tabular}{|P{4.5cm}|P{4.5cm}|P{5cm}|}
\hline
\centering
%and here 

%\cite{10.1007/978-3-030-78086-9_4}

\rowcolor{lightgray} \raggedright \textbf{W14:} Thesis writing, specification and homomorphic Eigenfaces continued. & Execute the second iteration of the architectural specification. Implement homomorphic eigen shift procedure, Goldschmidt's algorithm and write the sections "Homomorphic eigen shift procedure" and "Goldschmidt's algorithm" .  &\textbf{D28:} Second draft of the architectural specification. (A) \textbf{D29:} Homomorphic eigen shift procedure code. (H) \textbf{D30:} Homomorphic eigen shift procedure section. (H) \textbf{D31:} Homomorphic Goldschmidt's algorithm code. (H) 
\textbf{D32:} Homomorphic Goldschmidt's algorithm section. (H)\\ \hline

\rowcolor{gray} \textbf{W15:} & \centering \textbf{Break / catch-up.} & \\ \hline

\textbf{W16:} Thesis writing, specification and homomorphic Eigenfaces. & Execute the second iteration of the test specification. Implement HPCA, homomorphic facial recognition, vector operations and write the sections: "HPCA", "Homomorphic facial recognition" and "Vector operations".  & \textbf{D33:} Second draft of the test specification. (A) \textbf{D34:} HPCA code. (H) \textbf{D35:} HPCA section. (H) \textbf{D36:} Homomorphic facial recognition code. (H) \textbf{D37:} Homomorphic facial recognition section. (H) \textbf{D38:} Vector operations. (H)\\ \hline

\raggedright \textbf{W17:} Thesis writing and homomorphic Eigenfaces implementation in web application. & \raggedright Implement homomorphic Eigenfaces in the web application. Write the section: "Homomorphic Eigenfaces implementation" and adjust previous sections.  & \textbf{D39:} Homomorphic Eigenfaces implemented into the web application. (A) \textbf{D40:} Homomorphic Eigenfaces implementation (H)\\ \hline

\textbf{W18:} Thesis writing and specification. & \raggedright Finalize the requirements and architectural specification. Start on the results section. & \textbf{D41:} Final requirements specification. (A) \textbf{D42:} Final architectural specification. (A) \\ \hline

\rowcolor{lightgray} \textbf{W19:} Thesis writing and specification. & Finalize the test specification and continue on the results section. Start on the conclusive remarks section. & \textbf{D43:} Final test specification. (A) \\ \hline

\textbf{W20:} Thesis writing. & Write the rest of the implementation section of the thesis. & \textbf{D44:} Implementation. (A and H) \\ \hline

\textbf{W21:} Thesis writing. & Write the rest of the results section of the thesis and the conclusive remarks section.  & \textbf{D45:} Results. (A and H) \\ \hline
\rowcolor{lightgray}

\textbf{W22:} Thesis writing. & Write the abstract and the sections: "conclusive remarks" and "future work" of the thesis. & \textbf{D46:} Conclusive remarks. (A and H) 
\textbf{D47:} Abstract. (A and H)
\textbf{D48:} Future work. (A and H)
 \\ \hline

\textbf{W23:} Thesis writing. &  Focus on feedback and transcript the previous sections. &  
 \\ \hline
\rowcolor{gray} & \centering \textbf{Hand-in} & \\ \hline
%\cellcolor{gray}

\end{tabular}
%\caption{\label{}}
\end{table}
\end{flushleft}

\printbibliography

\end{document}
