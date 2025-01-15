// src/components/Questionnaire.js

import React, { useState } from 'react';

const Questionnaire = () => {
  const [answers, setAnswers] = useState({
    age: '',
    painLevel: '',
    stiffnessLevel: '',
    walkingDifficulty: ''
  });

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setAnswers((prevAnswers) => ({
      ...prevAnswers,
      [name]: value
    }));
  }

  const handleSubmit = (e) => {
    e.preventDefault();
    console.log('Questionnaire answers:', answers);
    // Handle the logic for submitting questionnaire answers (send to backend)
  }

  return (
    <div>
      <h2>Osteoarthritis Questionnaire</h2>
      <form onSubmit={handleSubmit}>
        <label>Age:</label>
        <input
          type="number"
          name="age"
          value={answers.age}
          onChange={handleInputChange}
          required
        />
        <br />
        <label>Pain Level (1-10):</label>
        <input
          type="number"
          name="painLevel"
          value={answers.painLevel}
          onChange={handleInputChange}
          required
        />
        <br />
        <label>Stiffness Level (1-10):</label>
        <input
          type="number"
          name="stiffnessLevel"
          value={answers.stiffnessLevel}
          onChange={handleInputChange}
          required
        />
        <br />
        <label>Walking Difficulty (1-10):</label>
        <input
          type="number"
          name="walkingDifficulty"
          value={answers.walkingDifficulty}
          onChange={handleInputChange}
          required
        />
        <br />
        <button type="submit">Submit Questionnaire</button>
      </form>
    </div>
  );
}

export default Questionnaire;
