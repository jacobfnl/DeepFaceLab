import { combineReducers } from "redux";
import CurrentArchitectureReducer from "./currentArchitecture/currentArchitectureReducer";

const rootReducer = combineReducers({
  currentArchitecture: CurrentArchitectureReducer,
});

export default rootReducer;
