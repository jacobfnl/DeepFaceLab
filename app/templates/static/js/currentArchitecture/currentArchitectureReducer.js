import { CurrentArchitectureActions } from '../constants/currentArchitectureConstants';

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