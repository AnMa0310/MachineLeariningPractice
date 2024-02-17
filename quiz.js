const quizContainer = document.getElementById('quiz');
const resultsContainer = document.getElementById('results');
const submitButton = document.getElementById('submit');

const myQuestions = [
  {
    question: "日本の首都はどこですか？",
    answers: {
      a: "大阪",
      b: "東京",
      c: "福岡"
    },
    correctAnswer: "b"
  },
  {
    question: "HTMLの略語は何ですか？",
    answers: {
      a: "Hyper Trainer Markup Language",
      b: "Hyper Text Markup Language",
      c: "Home Tool Markup Language"
    },
    correctAnswer: "b"
  },
  {
    question: "次のうち、最も軽い金属はどれですか？",
    answers: {
      a: "アルミニウム",
      b: "リチウム",
      c: "鉄"
    },
    correctAnswer: "b"
  }
];

function buildQuiz(){
  const output = [];

  myQuestions.forEach(
    (currentQuestion, questionNumber) => {
      const answers = [];

      for(letter in currentQuestion.answers){
        answers.push(
          `<label>
            <input type="radio" name="question${questionNumber}" value="${letter}">
            ${letter} :
            ${currentQuestion.answers[letter]}
          </label>`
        );
      }

      output.push(
        `<div class="question"> ${currentQuestion.question} </div>
        <div class="answers"> ${answers.join('')} </div>`
      );
    }
  );

  quizContainer.innerHTML = output.join('');
}

function showResults(){
  const answerContainers = quizContainer.querySelectorAll('.answers');

  let numCorrect = 0;

  myQuestions.forEach( (currentQuestion, questionNumber) => {
    const answerContainer = answerContainers[questionNumber];
    const selector = `input[name=question${questionNumber}]:checked`;
    const userAnswer = (answerContainer.querySelector(selector) || {}).value;

    if(userAnswer === currentQuestion.correctAnswer){
      numCorrect++;
      answerContainers[questionNumber].style.color = 'lightgreen';
    } else {
      answerContainers[questionNumber].style.color = 'red';
    }
  });

  resultsContainer.innerHTML = `${numCorrect} / ${myQuestions.length} 正解`;
}

buildQuiz();

submitButton.addEventListener('click', showResults);
