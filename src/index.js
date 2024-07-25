import React from "react";
import ReactDOM from "react-dom";
import axios from "axios";
import { API } from "./utils/constants";
import Api from "./utils/api";
import App from "./App";

const httpClient = axios.create({
  baseURL: API.baseURL
});

const api = new Api(httpClient);

ReactDOM.render(
  <React.StrictMode>
    <App api={api} />
  </React.StrictMode>,
  document.getElementById("root")
);
