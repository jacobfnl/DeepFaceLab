import { CurrentArchitectureActions } from '../constants/currentArchitectureConstants';
// https://medium.com/@pie6k/better-way-to-create-type-safe-redux-actions-and-reducers-with-typescript-45386808c103


export default function formStateReducer(state = null, action) {
const {data} = action;
  switch (action.type) {
    case CurrentArchitectureActions.SET_CURRENT_ARCHITECTURE: 
        // Overwrites current architecture with string submitted via action
        return data
    default:
      return state;
  }
}