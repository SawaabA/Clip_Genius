function createWindowMockUp(video) {
    return `<div class="mockup-browser border border-base-300 w-full">
                <div class="mockup-browser-toolbar">
                    <div class="input">https://jashandeep.co.uk</div>
                    </div>
                <div class="grid place-content-center h-80">${video}</div>
            </div>`

}


document.querySelectorAll("video").forEach(element => {
    console.log(element)
});



function CreatePresentation() {

    const h1 = "font-title relative z-2 mx-auto text-7xl lg:text-8xl leading-none font-black will-change-auto motion-reduce:transform-none! mb-20"
    const h2 = "text-[clamp(2.5rem,6vw,4.5rem)] leading-none font-light mt-[35%] mb-10 snap-start"
    const h3 = "text-4xl leading-none font-light mt-20 mb-5 snap-start"
    const h4 = "text-2xl leading-none font-light mt-20 mb-5 snap-start"
    const p = "text-base-content/70 font-title font-light md:text-2xl mb-6 mt-2 text-justify"
    const list = "grid grid-cols-1 lg:grid-cols-2 gap-20"
    const ul = "list-disc lg:pl-10 mb-10"
    const ol = "list-decimal lg:pl-10 mb-10"
    const li = "text-base-content/70 font-title font-light md:text-lg text-justify"

    updateCSS(['h1'], h1.split(" "));
    updateCSS(['h2'], h2.split(" "));
    updateCSS(['h3'], h3.split(" "));
    updateCSS(['h4'], h4.split(" "));
    updateCSS(['p'], p.split(" "));
    updateCSS(["ol"], list.split(" "));
    updateCSS(['ul'], ul.split(" "));
    updateCSS(['ol'], ol.split(" "));
    updateCSS(['li'], li.split(" "));

    updateCSS(['code'], ['bg-blue-50', 'p-1', 'rounded', 'font-bold', 'text-sm']);
    updateCSS(['pre'], ['bg-gray-100', 'p-4', 'rounded', 'overflow-x-auto', "mb-4"]);

    updateCSS(['table'], ['table', 'table-sm', 'mb-4', "mx-auto"]);
    updateCSS(['thead'], ['bg-gray-100']);
    updateCSS(['th'], ['px-4', 'py-2', 'text-left', 'font-bold']);
    updateCSS(['td'], ['px-4', 'py-2', 'border', 'border-gray-200']);
    updateCSS(['img'], ['p-1', "rounded-lg"]);

    updateCSS(['a'], ['text-blue-500', "hover:underline"]);
    updateCSS(['figcaption'], ['font-bold', "text-center", "mb-2"]);
    updateCSS(['hr'], ['my-6', 'border-t', 'border-gray-200']);
}

function updateIndex() {
    const index = document.getElementById("index")
    index.classList.add("hidden", "lg:grid", "lg:fixed", "h-screen", "bg-transparent", "opacity-5", "hover:opacity-100", "duration-200", "border-0",)
    // console.log(index)


    const main = document.querySelector("main")
    index.addEventListener("mouseenter", () => {
        main.style.opacity = "5%"
    })

    index.addEventListener("mouseleave", () => {
        main.style.opacity = "100%"
    })
}

loadScript("https://cdn.tailwindcss.com")
loadStylesheet("https://cdn.jsdelivr.net/npm/daisyui@4.12.23/dist/full.min.css");
document.documentElement.setAttribute("data-theme", "corporate");
setTimeout(() => {
    // updateCSS_ALL()
    generateIndex()
    CreatePresentation()
    updateIndex()
}, 200);

document.querySelector("main").className = "container mx-auto p-5 lg:p-20 transition duration-500 lg:translate-x-20 overflow-y-scroll snap-y snap-mandatory h-screen"

