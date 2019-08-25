import CurrentArchitectureActions from "../constants/currentArchitectureConstants";

// action creators

// Sets the current selected architecture

// Dummy data for currentArchitecture
const currentArchitecture = "liae";

export default {
	type: CurrentArchitectureActions.SET_CURRENT_ARCHITECTURE,
	data: currentArchitecture
};
