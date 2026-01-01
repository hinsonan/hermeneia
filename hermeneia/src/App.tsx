import { Component } from "solid-js";
import { Router, Route } from "@solidjs/router";
import Home from "./pages/Home";
import AudioEditor from "./pages/AudioEditor";
import "./styles/global.css";

const App: Component = () => {
  return (
    <Router>
      <Route path="/" component={Home} />
      <Route path="/audio-editor" component={AudioEditor} />
    </Router>
  );
};

export default App;
