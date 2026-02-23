import { create } from "zustand";
import { persist } from "zustand/middleware";

const applyTheme = (theme) => {
  document.documentElement.setAttribute("data-theme", theme);
};

const useThemeStore = create(
  persist(
    (set, get) => ({
      theme: "light",

      toggle: () => {
        const next = get().theme === "light" ? "dark" : "light";
        applyTheme(next);
        set({ theme: next });
      },

      hydrate: () => {
        applyTheme(get().theme);
      },
    }),
    {
      name: "theme-storage",
      partialize: (state) => ({ theme: state.theme }),
    }
  )
);

export default useThemeStore;
