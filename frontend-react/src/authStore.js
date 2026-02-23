import { create } from "zustand";
import { persist } from "zustand/middleware";
import { loginUser as apiLogin, registerUser as apiRegister } from "./api";

const useAuthStore = create(
  persist(
    (set) => ({
      user: null,
      isAuthenticated: false,

      login: async (userid, password) => {
        const data = await apiLogin(userid, password);
        if (data.success) {
          set({ user: { name: data.name, userid }, isAuthenticated: true });
          return { success: true };
        }
        return { success: false, error: "Invalid user ID or password" };
      },

      signup: async (name, userid, password) => {
        await apiRegister(name, userid, password);
        return { success: true };
      },

      logout: () => {
        set({ user: null, isAuthenticated: false });
      },
    }),
    {
      name: "auth-storage",
      partialize: (state) => ({
        user: state.user,
        isAuthenticated: state.isAuthenticated,
      }),
    }
  )
);

export default useAuthStore;
