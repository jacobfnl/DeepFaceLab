import React from "react";
import { Provider } from "react-redux";
import Store from "../store";

const Home = () => (
	<Provider store={Store}>
		<h1>Let the face-swapping begin!</h1>
	</Provider>
);

export default Home;
