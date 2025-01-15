// src/App.js

import React from 'react';
import { BrowserRouter as Router, Route, Switch } from 'react-router-dom';
import Home from './components/Home';
import Login from './components/Login';
import Signup from './components/Signup';
import Questionnaire from './components/Questionnaire';
import Prediction from './components/Prediction';

const App = () => {
  return (
    <Router>
      <div>
        <h1>Osteoarthritis Prediction App</h1>
        <Switch>
          <Route path="/" exact component={Home} />
          <Route path="/login" component={Login} />
          <Route path="/signup" component={Signup} />
          <Route path="/questionnaire" component={Questionnaire} />
          <Route path="/prediction" component={Prediction} />
        </Switch>
      </div>
    </Router>
  );
}

export default App;
