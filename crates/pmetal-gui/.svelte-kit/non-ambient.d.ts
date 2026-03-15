
// this file is generated — do not edit it


declare module "svelte/elements" {
	export interface HTMLAttributes<T> {
		'data-sveltekit-keepfocus'?: true | '' | 'off' | undefined | null;
		'data-sveltekit-noscroll'?: true | '' | 'off' | undefined | null;
		'data-sveltekit-preload-code'?:
			| true
			| ''
			| 'eager'
			| 'viewport'
			| 'hover'
			| 'tap'
			| 'off'
			| undefined
			| null;
		'data-sveltekit-preload-data'?: true | '' | 'hover' | 'tap' | 'off' | undefined | null;
		'data-sveltekit-reload'?: true | '' | 'off' | undefined | null;
		'data-sveltekit-replacestate'?: true | '' | 'off' | undefined | null;
	}
}

export {};


declare module "$app/types" {
	type MatcherParam<M> = M extends (param : string) => param is (infer U extends string) ? U : string;

	export interface AppTypes {
		RouteId(): "/" | "/datasets" | "/distillation" | "/grpo" | "/inference" | "/merging" | "/models" | "/quantize" | "/settings" | "/training";
		RouteParams(): {
			
		};
		LayoutParams(): {
			"/": Record<string, never>;
			"/datasets": Record<string, never>;
			"/distillation": Record<string, never>;
			"/grpo": Record<string, never>;
			"/inference": Record<string, never>;
			"/merging": Record<string, never>;
			"/models": Record<string, never>;
			"/quantize": Record<string, never>;
			"/settings": Record<string, never>;
			"/training": Record<string, never>
		};
		Pathname(): "/" | "/datasets" | "/distillation" | "/grpo" | "/inference" | "/merging" | "/models" | "/quantize" | "/settings" | "/training";
		ResolvedPathname(): `${"" | `/${string}`}${ReturnType<AppTypes['Pathname']>}`;
		Asset(): string & {};
	}
}