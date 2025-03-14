/**
-------------------------------------------------------
Report Formatter: Enhances plain HTML reports by embedding Tailwind CSS and DaisyUI components
-------------------------------------------------------
Author:  JD
ID:      91786
Uses:    DOM Manipulation
Version: 1.0.8
__updated__ = Fri Mar 14 2025
-------------------------------------------------------*/

function generateIndex(containerId = "index") {
    const headings = document.querySelectorAll("h2,h3, h4, h5, h6");
    if (!headings.length) {
        console.warn("No headings found in the document.");
        return;
    }

    // Create the container for the index
    let container = document.getElementById(containerId);
    if (!container) {
        container = document.createElement("div");
        container.id = containerId;
        document.body.insertBefore(container, document.body.firstChild);
    }
    container.innerHTML = `<h3 style='margin-top:15px;'>Table of Contents</h3>`;
    const indexList = document.createElement("ul");
    indexList.className = "menu"
    container.appendChild(indexList);

    headings.forEach(heading => {
        const level = parseInt(heading.tagName[1]);
        const listItem = document.createElement("li");

        if (!heading.id) {
            heading.id = `91oz.-${Math.random().toString(36).substr(2, 9)}`;
        }
        const link = document.createElement("a");
        link.href = `#${heading.id}`;
        // link.className = "m-0 p-0 link"
        link.textContent = heading.textContent;
        listItem.appendChild(link);
        listItem.style.marginLeft = `${(level - 2) * 20}px`;

        indexList.appendChild(listItem);
    });
    container.classList.add("lg:w-64", "overflow-y-scroll", "border", "print:border-0", "p-3")
}

function updateCSS(tags, addClasses = [], removeClasses = []) {
    tags.forEach(tag => {
        document.querySelectorAll(tag).forEach(element => {
            removeClasses.forEach(cls => element.classList.remove(cls));
            addClasses.forEach(cls => element.classList.add(cls));
        });
    });
}

function loadScript(src) {
    if (!document.querySelector(`script[src="${src}"]`)) {
        let script = document.createElement("script");
        script.src = src;
        script.onerror = () => console.error(`Failed to load script: ${src}`);
        document.head.appendChild(script);
    } else {
        console.log(`Script already loaded: ${src}`);
    }
}

// Function to dynamically load a stylesheet
function loadStylesheet(href) {
    if (!document.querySelector(`link[href="${href}"]`)) {
        let link = document.createElement("link");
        link.rel = "stylesheet";
        link.href = href;
        document.head.appendChild(link);
        console.log(`Stylesheet loaded: ${href}`);
    } else {
        console.log(`Stylesheet already loaded: ${href}`);
    }
}

function updateCSS_ALL() {
    updateCSS(['h1'], ['text-3xl', 'font-bold']);
    updateCSS(['h2'], ['text-2xl', 'font-semibold', 'mt-8']);
    updateCSS(['h3'], ['text-lg', 'font-semibold', 'mt-5']);
    updateCSS(['h4'], ['text-md', 'font-semibold', 'mt-4']);
    updateCSS(['h5'], ['text-sm', 'font-semibold', 'mt-3']);
    updateCSS(['h6'], ['text-xs', 'font-semibold', 'mt-2']);
    updateCSS(['p'], ['mt-1', 'text-md', 'text-muted-foreground', 'text-justify']);
    updateCSS(['ul'], ['list-disc', 'pl-10']);
    updateCSS(['ol'], ['list-decimal', 'pl-10']);
    updateCSS(['li'], ['m-0', 'p-0']);
    updateCSS(['table'], ['table', 'table-sm', 'mb-4']);
    updateCSS(['thead'], ['bg-gray-100']);
    updateCSS(['th'], ['px-4', 'py-2', 'text-left', 'font-bold']);
    updateCSS(['td'], ['px-4', 'py-2', 'border', 'border-gray-200']);
    updateCSS(['img'], ['p-10']);
    updateCSS(['blockquote'], ['border-l-4', 'border-gray-300', 'pl-4', 'italic', 'text-gray-600']);
    updateCSS(['code'], ['bg-gray-100', 'p-1', 'rounded', 'text-sm', 'font-mono']);
    updateCSS(['pre'], ['bg-gray-100', 'p-4', 'rounded', 'overflow-x-auto']);
    updateCSS(['hr'], ['my-6', 'border-t', 'border-gray-200']);
    updateCSS(['button'], ['px-4', 'py-2', 'bg-blue-500', 'text-white', 'rounded', 'hover:bg-blue-600']);
    updateCSS(['input'], ['px-3', 'py-2', 'border', 'border-gray-300', 'rounded', 'focus:outline-none', 'focus:ring-2', 'focus:ring-blue-500']);
    updateCSS(['textarea'], ['px-3', 'py-2', 'border', 'border-gray-300', 'rounded', 'focus:outline-none', 'focus:ring-2', 'focus:ring-blue-500']);
    updateCSS(['label'], ['block', 'text-sm', 'font-medium', 'text-gray-700']);
    updateCSS(['form'], ['space-y-4']);
    updateCSS(['section'], ['my-8']);
    updateCSS(['div'], ['my-4']);
}



loadScript("https://cdn.tailwindcss.com")
loadStylesheet("https://cdn.jsdelivr.net/npm/daisyui@4.12.23/dist/full.min.css");
document.documentElement.setAttribute("data-theme", "light");
setTimeout(() => {
    updateCSS_ALL()
    generateIndex()
}, 200);

document.querySelector("main").className = "container mx-auto px-4 py-8 overflow-y-scroll print:px-0"
document.body.classList.add("lg:flex", "print:block", "h-screen", "text-sm")
