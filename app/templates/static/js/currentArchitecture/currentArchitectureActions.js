import { CurrentArchitectureActions } from '../constants/currentArchitectureConstants';

    // action creators

    // Sets the current selected architecture
    export function setCurrentArchitecture(currentArchitecture) {
        return { type: CurrentArchitectureActions.SET_CURRENT_ARCHITECTURE, data: currentArchitecture };
    }
