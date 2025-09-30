// Space theme animations
document.addEventListener('DOMContentLoaded', function() {
    // Create shooting stars
    setInterval(createShootingStar, 3000);
    
    // Create floating planets
    createPlanets();
    
    // Parallax effect
    window.addEventListener('scroll', function() {
        const scrolled = window.pageYOffset;
        const parallaxElements = document.querySelectorAll('.parallax');
        
        parallaxElements.forEach(function(el) {
            const speed = parseFloat(el.getAttribute('data-speed')) || 0.5;
            el.style.transform = `translateY(${scrolled * speed}px)`;
        });
    });
});

function createShootingStar() {
    const container = document.querySelector('.stars');
    const star = document.createElement('div');
    star.className = 'shooting-star';
    
    // Random position
    const startX = Math.random() * window.innerWidth;
    const startY = Math.random() * 100;
    
    // Random size
    const size = Math.random() * 3 + 1;
    
    star.style.left = `${startX}px`;
    star.style.top = `${startY}px`;
    star.style.width = `${size}px`;
    star.style.height = `${size}px`;
    
    container.appendChild(star);
    
    // Remove after animation
    setTimeout(() => {
        star.remove();
    }, 2000);
}

function createPlanets() {
    const container = document.querySelector('.twinkling');
    const planets = [
        { color: '#FF6B6B', size: 40, orbit: 100, speed: 40 },
        { color: '#4ECDC4', size: 30, orbit: 150, speed: 30 },
        { color: '#FFE66D', size: 20, orbit: 200, speed: 20 }
    ];
    
    planets.forEach(planet => {
        const planetEl = document.createElement('div');
        planetEl.className = 'planet';
        planetEl.style.width = `${planet.size}px`;
        planetEl.style.height = `${planet.size}px`;
        planetEl.style.background = `radial-gradient(circle at 30% 30%, white, ${planet.color})`;
        planetEl.style.boxShadow = `0 0 20px ${planet.color}`;
        
        // Create orbit
        const orbitEl = document.createElement('div');
        orbitEl.className = 'orbit';
        orbitEl.style.width = `${planet.orbit * 2}px`;
        orbitEl.style.height = `${planet.orbit * 2}px`;
        orbitEl.style.animationDuration = `${planet.speed}s`;
        
        orbitEl.appendChild(planetEl);
        container.appendChild(orbitEl);
    });
}

// Game-specific interactivity
function initDrawingCanvas() {
    const canvas = document.getElementById('drawing-canvas');
    const ctx = canvas.getContext('2d');
    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;
    
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);
    
    // Touch events for mobile
    canvas.addEventListener('touchstart', handleTouch);
    canvas.addEventListener('touchmove', handleTouch);
    canvas.addEventListener('touchend', stopDrawing);
    
    function startDrawing(e) {
        isDrawing = true;
        [lastX, lastY] = [e.offsetX, e.offsetY];
    }
    
    function draw(e) {
        if (!isDrawing) return;
        
        ctx.beginPath();
        ctx.lineWidth = 5;
        ctx.lineCap = 'round';
        ctx.strokeStyle = '#000';
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(e.offsetX, e.offsetY);
        ctx.stroke();
        
        [lastX, lastY] = [e.offsetX, e.offsetY];
    }
    
    function stopDrawing() {
        isDrawing = false;
    }
    
    function handleTouch(e) {
        e.preventDefault();
        const touch = e.touches[0];
        const mouseEvent = new MouseEvent(
            e.type === 'touchstart' ? 'mousedown' : 'mousemove',
            {
                clientX: touch.clientX,
                clientY: touch.clientY
            }
        );
        canvas.dispatchEvent(mouseEvent);
    }
}

// Voice assistant functions
async function speakText(text) {
    if ('speechSynthesis' in window) {
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 0.9;
        utterance.pitch = 1.2;
        window.speechSynthesis.speak(utterance);
    }
}

async function listenForSpeech() {
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
        const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
        recognition.continuous = false;
        recognition.interimResults = false;
        
        return new Promise((resolve, reject) => {
            recognition.onresult = function(event) {
                const transcript = event.results[0][0].transcript;
                resolve(transcript);
            };
            
            recognition.onerror = function(event) {
                reject(event.error);
            };
            
            recognition.start();
        });
    } else {
        throw new Error('Speech recognition not supported');
    }
}

// Drag and drop functionality for games
function initDragAndDrop() {
    const draggables = document.querySelectorAll('.draggable');
    const dropZones = document.querySelectorAll('.drop-zone');
    
    draggables.forEach(draggable => {
        draggable.addEventListener('dragstart', () => {
            draggable.classList.add('dragging');
        });
        
        draggable.addEventListener('dragend', () => {
            draggable.classList.remove('dragging');
        });
    });
    
    dropZones.forEach(zone => {
        zone.addEventListener('dragover', e => {
            e.preventDefault();
            zone.classList.add('drag-over');
        });
        
        zone.addEventListener('dragleave', () => {
            zone.classList.remove('drag-over');
        });
        
        zone.addEventListener('drop', e => {
            e.preventDefault();
            zone.classList.remove('drag-over');
            
            const draggable = document.querySelector('.dragging');
            if (draggable && zone.dataset.accepts === draggable.dataset.type) {
                zone.appendChild(draggable);
                draggable.classList.add('dropped');
                
                // Trigger custom event
                const event = new CustomEvent('itemDropped', {
                    detail: {
                        item: draggable.dataset.value,
                        zone: zone.id
                    }
                });
                document.dispatchEvent(event);
            }
        });
    });
}

// Initialize games when page loads
window.addEventListener('load', function() {
    // Check if we're on a game page and initialize accordingly
    if (document.getElementById('drawing-canvas')) {
        initDrawingCanvas();
    }
    
    if (document.querySelector('.draggable')) {
        initDragAndDrop();
    }
    
    // Add space theme to all pages
    document.body.classList.add('space-theme');
});