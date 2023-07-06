import React from 'react';
import { BrowserRouter as Router, Route, Switch } from 'react-router-dom';
import Login from './components/Login';
import SignUp from './components/SignUp';
import Logout from './components/Logout';
import ProtectedRoute from './components/ProtectedRoute';
import './styles/global.css';

const App: React.FC = () => {
  return (
    <Router>
      <Switch>
        <Route path="/login" component={Login} />
        <Route path="/signup" component={SignUp} />
        <Route path="/logout" component={Logout} />
        <ProtectedRoute path="/" component={Home} />
      </Switch>
    </Router>
  );
}

export default App;