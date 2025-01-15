// src/components/Home.js

import React from 'react';
import { Link } from 'react-router-dom';

const Home = () => {
  return (
    <div>
      <h1>Welcome to Osteoarthritis Prediction</h1>
      <p>Predict the severity of osteoarthritis based on a questionnaire and X-ray images.</p>
      <nav>
        <ul>
          <li>
            <Link to="/login">Login</Link>
          </li>
          <li>
            <Link to="/signup">Signup</Link>
          </li>
          <li>
            <Link to="/questionnaire">Fill out Questionnaire</Link>
          </li>
          <li>
            <Link to="/prediction">Prediction</Link>
          </li>
        </ul>
      </nav>
    </div>
  );
}

export default Home;
