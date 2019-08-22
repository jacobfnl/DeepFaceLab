import { createStore } from 'redux';
import rootReducer from './rootReducer';

export default (initialState) => createStore(rootReducer, initialState);
