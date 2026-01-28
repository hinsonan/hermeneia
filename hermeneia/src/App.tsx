import { Component } from "solid-js";
import { Router, Route } from "@solidjs/router";
import Home from "./pages/Home";
import AudioEditor from "./pages/AudioEditor";
import Transcription from "./pages/Transcription";
import Translation from "./pages/Translation";
import "./styles/global.css";

const App: Component = () => {
  return (
    <Router>
      <Route path="/" component={Home} />
      <Route path="/audio-editor" component={AudioEditor} />
      <Route path="/transcription" component={Transcription} />
      <Route path="/translation" component={Translation} />
    </Router>
  );
};

export default App;
