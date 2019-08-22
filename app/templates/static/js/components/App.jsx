import React, { Component } from 'react';
import { Provider } from "react-redux";
import Store from "./../store.js";

const StoreInstance = Store({});


export default class Home extends Component {
    render() {
       return (
         <Provider store={StoreInstance}>
             <h1>Let the face swapping begin!</h1>
         </Provider>
       )
    }
}