document.addEventListener('DOMContentLoaded', () => {
    // State Container
    const state = {
        currentStep: 1,
        totalSteps: 9,
        vehicleCatalog: [],
        formData: {
            full_name: '',
            phone_number: '',
            email_address: '',
            gender: 'Male',
            age: null,
            qualifications: 'GRAD',
            employment_type: null,
            resident_type: null,
            pincode: null,
            net_salary: null,
            past_loans_active: 'NO_PAST_LOANS',
            product_code: 'MC',
            make_code: null,
            vehicle_name: null,
            vehicle_price: null,
            loan_amount: null,
            ltv: null
        }
    };

    // DOM Elements
    const chatMessages = document.getElementById('chat-messages');
    const dynamicInputArea = document.getElementById('dynamic-input-area');
    const progressBarFill = document.getElementById('progress-bar-fill');
    const stepIndicator = document.getElementById('step-indicator');
    const stepPercentage = document.getElementById('step-percentage');
    const chatInputText = document.getElementById('chat-input-text');
    const btnSendChat = document.getElementById('btn-send-chat');

    const toggleAuditBtn = document.getElementById('toggle-audit-btn');
    const auditModal = document.getElementById('audit-modal');
    const closeAuditBtn = document.getElementById('close-audit-btn');
    const auditLogsContainer = document.getElementById('audit-logs-container');

    // Fetch Sanitized Vehicle Catalog from CSV
    async function loadVehicleCatalog() {
        try {
            const res = await fetch('/api/vehicle-catalog');
            state.vehicleCatalog = await res.json();
        } catch (err) {
            console.error('Failed to load vehicle catalog:', err);
        }
    }

    // Fetch Health Status & Groq LLM Connectivity
    async function checkHealthStatus() {
        try {
            const res = await fetch('/api/health');
            const data = await res.json();
            const badgeText = document.getElementById('groq-status-text');
            const sidebarText = document.getElementById('sidebar-groq-status');
            
            if (data.groq_connected) {
                if (badgeText) badgeText.innerText = "⚡ Groq LLM Connected";
                if (sidebarText) sidebarText.innerText = "⚡ Connected (Llama-3.1-8b)";
            } else {
                if (badgeText) badgeText.innerText = "⚠️ Groq LLM Offline";
                if (sidebarText) sidebarText.innerText = "⚠️ Offline Fallback";
            }
        } catch (err) {
            console.error('Failed health check:', err);
        }
    }

    // Initialize Chat
    async function initChat() {
        await loadVehicleCatalog();
        await checkHealthStatus();
        appendBotMessage("<p>Hello! Welcome to <strong>ABC Credit</strong>. I am your instant AI loan approval assistant. 🚀</p><p>Answer quick questions or type naturally in the chatbox below for an instant pre-approval decision!</p>");
        renderStepInput(1);
    }
    initChat();

    function updateProgress(step) {
        state.currentStep = step;
        const pct = Math.round((step / state.totalSteps) * 100);
        progressBarFill.style.width = `${pct}%`;
        stepPercentage.innerText = `${pct}%`;
        stepIndicator.innerText = `Step ${step} of ${state.totalSteps}`;
    }

    function appendUserMessage(text) {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message message-user animate-fade-in';
        msgDiv.innerHTML = `
            <div class="avatar avatar-user"><i class="fa-solid fa-user"></i></div>
            <div class="message-content"><p>${text}</p></div>
        `;
        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function appendBotMessage(textHtml) {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message message-bot animate-fade-in';
        msgDiv.innerHTML = `
            <div class="avatar avatar-bot"><i class="fa-solid fa-robot"></i></div>
            <div class="message-content">
                <div class="bot-header">
                    <span class="bot-name">ABC Credit Assistant</span>
                    <span class="bot-time">${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                </div>
                ${textHtml}
            </div>
        `;
        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Single-Input Architecture: Step cards render options/badges, while ONLY ONE chat input field exists at the bottom!
    function renderStepInput(step) {
        updateProgress(step);
        dynamicInputArea.innerHTML = '';

        const container = document.createElement('div');
        container.className = 'step-box';

        switch (step) {
            case 1:
                chatInputText.placeholder = "Type your full name and gender (e.g. 'I am Rahul, Male')...";
                container.innerHTML = `
                    <div class="step-question">
                        <span class="question-num">1</span>
                        <span>Applicant Name & Gender</span>
                    </div>
                    <div class="pills-grid" id="pills-gender" style="margin-top:6px;">
                        <button class="pill-btn ${state.formData.gender === 'Male' ? 'active' : ''}" data-val="Male"><i class="fa-solid fa-mars"></i> Male</button>
                        <button class="pill-btn ${state.formData.gender === 'Female' ? 'active' : ''}" data-val="Female"><i class="fa-solid fa-venus"></i> Female</button>
                    </div>
                    <div class="nl-nudge-banner" style="margin-top:8px;"><i class="fa-solid fa-keyboard text-amber"></i> Please type your full name in the single chatbox below!</div>
                `;
                dynamicInputArea.appendChild(container);
                setupPills('pills-gender', (val) => { state.formData.gender = val; });
                break;

            case 2: {
                chatInputText.placeholder = "Type vehicle model (e.g. Suzuki Gixxer, Hayabusa, TVS Jupiter)...";
                let dropdownOptionsHtml = `<option value="">-- Select from Sanitized TVS Dataset Catalog --</option>`;
                state.vehicleCatalog.forEach(item => {
                    const isSel = (state.formData.vehicle_name === item.model_description) ? 'selected' : '';
                    dropdownOptionsHtml += `<option value="${item.model_description}" data-make="${item.make_code}" data-price="${item.typical_price}" ${isSel}>${item.model_description} (₹${item.typical_price.toLocaleString()})</option>`;
                });
                dropdownOptionsHtml += `<option value="OTHERS">Others / Custom Brand (Type model in chatbox below)</option>`;

                container.innerHTML = `
                    <div class="step-question">
                        <span class="question-num">2</span>
                        <span>Select Vehicle Model</span>
                    </div>
                    <div style="font-size:12px; color:var(--text-muted); margin-bottom:6px;">Select Vehicle Description from Training Dataset:</div>
                    <select id="select-vehicle-csv" class="form-control" style="width:100%; margin-bottom:10px; background:var(--bg-card); color:var(--text-main); font-size:13px;">
                        ${dropdownOptionsHtml}
                    </select>
                    <div class="nl-nudge-banner"><i class="fa-solid fa-keyboard text-amber"></i> Or simply type your vehicle model in the chatbox below!</div>
                `;
                dynamicInputArea.appendChild(container);

                const selectVeh = document.getElementById('select-vehicle-csv');
                selectVeh.addEventListener('change', async (e) => {
                    const val = e.target.value;
                    if (val && val !== 'OTHERS') {
                        chatInputText.value = val;
                        await handleConversationalTurn();
                    }
                });
                break;
            }

            case 3: {
                const vehName = state.formData.vehicle_name || state.formData.make_code || 'Vehicle';
                const predPrice = state.formData.suggested_price || state.formData.vehicle_price || null;
                chatInputText.placeholder = `Type negotiated on-road deal price for ${vehName} in ₹ (e.g. 112500)...`;

                let predPriceBtnHtml = '';
                if (predPrice && predPrice > 0) {
                    predPriceBtnHtml = `
                        <div style="font-size:12px; color:var(--text-muted); margin-top:8px; margin-bottom:6px;">Estimated On-Road Deal Price (Click to Quick Submit):</div>
                        <div class="pills-grid" id="pills-predicted-price" style="margin-bottom:10px;">
                            <button class="pill-btn active" data-val="${predPrice}" style="background:rgba(16,185,129,0.15); border:1.5px solid var(--emerald); color:var(--text-main); font-weight:700; padding:12px 18px; font-size:14px;">
                                <i class="fa-solid fa-tag text-emerald"></i> ₹${Number(predPrice).toLocaleString('en-IN')} (Estimated for ${vehName})
                            </button>
                        </div>
                    `;
                }

                container.innerHTML = `
                    <div class="step-question">
                        <span class="question-num">3</span>
                        <span>Negotiated Vehicle On-Road Price</span>
                    </div>
                    <div style="font-size:12px; color:var(--text-muted); margin-bottom:6px;">
                        Selected Vehicle: <strong>${vehName}</strong>
                    </div>
                    ${predPriceBtnHtml}
                    <div class="nl-nudge-banner"><i class="fa-solid fa-keyboard text-amber"></i> Click the estimated price button above or type your exact price in the chatbox below!</div>
                `;
                dynamicInputArea.appendChild(container);

                if (predPrice && predPrice > 0) {
                    setupPills('pills-predicted-price', async (val) => {
                        chatInputText.value = val;
                        await handleConversationalTurn();
                    });
                }
                break;
            }

            case 4: {
                const vehName = state.formData.vehicle_name || state.formData.make_code || 'Vehicle';
                const vehPrice = state.formData.vehicle_price || 110000;
                const currentLoan = state.formData.loan_amount || null;
                const currentLtv = currentLoan ? ((currentLoan / vehPrice) * 100).toFixed(1) : null;
                chatInputText.placeholder = `Type requested loan amount for ${vehName} in ₹ (e.g. 85000)...`;

                const ltvOptions = [40, 50, 60, 70];

                container.innerHTML = `
                    <div class="step-question">
                        <span class="question-num">4</span>
                        <span>Requested Loan Amount &amp; LTV</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; font-size:12px; background:rgba(255,255,255,0.05); padding:10px 14px; border-radius:8px; margin-bottom:10px;">
                        <span>Vehicle Price: <strong>₹${vehPrice.toLocaleString()}</strong></span>
                        <span id="ltv-live-display">Calculated LTV: <strong class="text-indigo">${currentLtv ? currentLtv + '%' : '—'}</strong></span>
                    </div>
                    <div style="font-size:12px; color:var(--text-muted); margin-bottom:8px;">Quick LTV Select — pick % of vehicle price as loan:</div>
                    <div class="pills-grid" id="pills-ltv-pct" style="grid-template-columns: repeat(4, 1fr); gap:8px; margin-bottom:10px;">
                        ${ltvOptions.map(pct => {
                            const loanVal = Math.round(vehPrice * pct / 100);
                            const isActive = currentLoan && Math.abs(currentLoan - loanVal) < 100;
                            return `<button class="pill-btn${isActive ? ' active' : ''}" data-pct="${pct}" data-loan="${loanVal}" style="flex-direction:column; gap:2px; padding:10px 6px;">
                                <span style="font-size:15px; font-weight:700;">${pct}%</span>
                                <span style="font-size:10px; color:var(--text-muted);">₹${loanVal.toLocaleString()}</span>
                            </button>`;
                        }).join('')}
                    </div>
                    <div class="nl-nudge-banner"><i class="fa-solid fa-keyboard text-amber"></i> Or type any custom loan amount in the chatbox below!</div>
                `;
                dynamicInputArea.appendChild(container);

                // Wire up LTV% buttons
                document.getElementById('pills-ltv-pct').querySelectorAll('.pill-btn').forEach(btn => {
                    btn.addEventListener('click', async () => {
                        const pct = parseInt(btn.dataset.pct);
                        const loanVal = parseInt(btn.dataset.loan);
                        const ltv = pct.toFixed(1);

                        // Highlight active button
                        document.querySelectorAll('#pills-ltv-pct .pill-btn').forEach(b => b.classList.remove('active'));
                        btn.classList.add('active');

                        // Update live LTV display in card
                        document.getElementById('ltv-live-display').innerHTML =
                            `Calculated LTV: <strong class="text-indigo">${ltv}%</strong>`;

                        state.formData.loan_amount = loanVal;
                        state.formData.ltv = pct;

                        // Set value in chatbox so handleConversationalTurn picks it up & sends to Groq
                        chatInputText.value = `${loanVal}`;
                        await handleConversationalTurn();
                    });
                });
                break;
            }

            case 5:
                chatInputText.placeholder = "Type your net monthly income in ₹ (e.g. 45000)...";
                container.innerHTML = `
                    <div class="step-question">
                        <span class="question-num">5</span>
                        <span>Monthly Net Salary / Income</span>
                    </div>
                    <div style="font-size:12px; color:var(--text-muted); margin-bottom:6px;">Select Quick Income Preset or Type in Chatbox:</div>
                    <div class="pills-grid" id="pills-salary-preset">
                        <button class="pill-btn" data-val="25000">₹25,000</button>
                        <button class="pill-btn" data-val="35000">₹35,000</button>
                        <button class="pill-btn active" data-val="45000">₹45,000</button>
                        <button class="pill-btn" data-val="75000">₹75,000</button>
                        <button class="pill-btn" data-val="120000">₹1,20,000</button>
                    </div>
                    <div class="nl-nudge-banner" style="margin-top:8px;"><i class="fa-solid fa-keyboard text-amber"></i> Or simply type your monthly income in the chatbox below!</div>
                `;
                dynamicInputArea.appendChild(container);

                setupPills('pills-salary-preset', async (val) => {
                    chatInputText.value = val;
                    await handleConversationalTurn();
                });
                break;

            case 6:
                chatInputText.placeholder = "Type employment sector (e.g. salaried, freelancer, farmer, shopkeeper)...";
                container.innerHTML = `
                    <div class="step-question">
                        <span class="question-num">6</span>
                        <span>Employment Sector / Occupation</span>
                    </div>
                    <div style="font-size:12px; color:var(--text-muted); margin-bottom:8px;">Select Employment Category:</div>
                    <div class="pills-grid" id="pills-emp" style="grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));">
                        <button class="pill-btn ${state.formData.employment_type === 'SAL' ? 'active' : ''}" data-val="SAL"><i class="fa-solid fa-briefcase"></i> Salaried Employee</button>
                        <button class="pill-btn ${state.formData.employment_type === 'SEP' ? 'active' : ''}" data-val="SEP"><i class="fa-solid fa-user-tie"></i> Self-Employed Prof</button>
                        <button class="pill-btn ${state.formData.employment_type === 'AGR' ? 'active' : ''}" data-val="AGR"><i class="fa-solid fa-wheat-awn"></i> Agriculture / Farmer</button>
                        <button class="pill-btn ${state.formData.employment_type === 'NREGI' ? 'active' : ''}" data-val="NREGI"><i class="fa-solid fa-store"></i> Shopkeeper / Trader</button>
                        <button class="pill-btn ${state.formData.employment_type === 'STU' ? 'active' : ''}" data-val="STU"><i class="fa-solid fa-graduation-cap"></i> Student</button>
                        <button class="pill-btn ${state.formData.employment_type === 'NPP' ? 'active' : ''}" data-val="NPP"><i class="fa-solid fa-laptop-code"></i> Freelancer / Private</button>
                        <button class="pill-btn ${state.formData.employment_type === 'PEN' ? 'active' : ''}" data-val="PEN"><i class="fa-solid fa-bed"></i> Pensioner / Retired</button>
                        <button class="pill-btn ${state.formData.employment_type === 'NONEARNMEM' ? 'active' : ''}" data-val="NONEARNMEM"><i class="fa-solid fa-house-user"></i> Homemaker / Non-Earning</button>
                    </div>
                    <div class="nl-nudge-banner" style="margin-top:8px;"><i class="fa-solid fa-keyboard text-amber"></i> Or simply type your employment status in the chatbox below!</div>
                `;
                dynamicInputArea.appendChild(container);

                setupPills('pills-emp', async (val) => {
                    chatInputText.value = val;
                    await handleConversationalTurn();
                });
                break;

            case 7:
                chatInputText.placeholder = "Type residential status (e.g. owned house, rented, company provided)...";
                container.innerHTML = `
                    <div class="step-question">
                        <span class="question-num">7</span>
                        <span>Residential Housing Status</span>
                    </div>
                    <div class="pills-grid" id="pills-res">
                        <button class="pill-btn ${state.formData.resident_type === 'O' ? 'active' : ''}" data-val="O"><i class="fa-solid fa-house"></i> Owned House</button>
                        <button class="pill-btn ${state.formData.resident_type === 'R' ? 'active' : ''}" data-val="R"><i class="fa-solid fa-building-user"></i> Rented</button>
                        <button class="pill-btn ${state.formData.resident_type === 'L' ? 'active' : ''}" data-val="L"><i class="fa-solid fa-key"></i> Leased</button>
                        <button class="pill-btn ${state.formData.resident_type === 'CO' ? 'active' : ''}" data-val="CO"><i class="fa-solid fa-building"></i> Company Provided</button>
                    </div>
                    <div class="nl-nudge-banner" style="margin-top:8px;"><i class="fa-solid fa-keyboard text-amber"></i> Or simply type your housing status in the chatbox below!</div>
                `;
                dynamicInputArea.appendChild(container);

                setupPills('pills-res', async (val) => {
                    chatInputText.value = val;
                    await handleConversationalTurn();
                });
                break;

            case 8:
                chatInputText.placeholder = "Type 6-digit residential Pincode (e.g. 500090)...";
                container.innerHTML = `
                    <div class="step-question">
                        <span class="question-num">8</span>
                        <span>Residential Pincode</span>
                    </div>
                    <div style="margin-top:8px; margin-bottom:10px;">
                        <button id="btn-gps-pin" class="pill-btn active" style="background:rgba(99,102,241,0.2); border:1px solid var(--primary); font-size:13px; height:42px;"><i class="fa-solid fa-location-crosshairs text-indigo"></i> Auto-Detect GPS Location</button>
                    </div>
                    <div style="font-size:12px; color:var(--text-muted); margin-bottom:6px;">Or Select Quick Pincode Preset:</div>
                    <div class="pills-grid" id="pills-pincode-preset">
                        <button class="pill-btn" data-val="560076">560076 (Bengaluru)</button>
                        <button class="pill-btn" data-val="500090">500090 (Hyderabad)</button>
                        <button class="pill-btn" data-val="110030">110030 (Delhi)</button>
                        <button class="pill-btn" data-val="400001">400001 (Mumbai)</button>
                        <button class="pill-btn" data-val="600001">600001 (Chennai)</button>
                    </div>
                    <div class="nl-nudge-banner" style="margin-top:8px;"><i class="fa-solid fa-keyboard text-amber"></i> Or type any 6-digit Pincode in the chatbox below!</div>
                `;
                dynamicInputArea.appendChild(container);

                // GPS Auto-Detect Pincode Button Listener
                const gpsBtn = document.getElementById('btn-gps-pin');
                gpsBtn.addEventListener('click', () => {
                    if (!navigator.geolocation) {
                        alert('Geolocation is not supported by your browser. Please select a preset or type your Pincode.');
                        return;
                    }
                    gpsBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Detecting GPS Location...`;
                    navigator.geolocation.getCurrentPosition(async (pos) => {
                        try {
                            const res = await fetch(`https://api.bigdatacloud.net/data/reverse-geocode-client?latitude=${pos.coords.latitude}&longitude=${pos.coords.longitude}&localityLanguage=en`);
                            const geo = await res.json();
                            const detectedPin = geo.postcode || '560076';
                            gpsBtn.innerHTML = `<i class="fa-solid fa-check text-emerald"></i> GPS Detected: ${detectedPin}`;
                            chatInputText.value = detectedPin;
                            await handleConversationalTurn();
                        } catch (err) {
                            gpsBtn.innerHTML = `<i class="fa-solid fa-location-dot"></i> Default: 560076`;
                            chatInputText.value = '560076';
                            await handleConversationalTurn();
                        }
                    }, (err) => {
                        gpsBtn.innerHTML = `<i class="fa-solid fa-location-dot"></i> Default: 560076`;
                        chatInputText.value = '560076';
                        handleConversationalTurn();
                    });
                });

                // Pincode Preset Pills Listener
                setupPills('pills-pincode-preset', async (val) => {
                    chatInputText.value = val;
                    await handleConversationalTurn();
                });
                break;

            case 9:
                chatInputText.placeholder = "Type your age in years (e.g. 32)...";
                container.innerHTML = `
                    <div class="step-question">
                        <span class="question-num">9</span>
                        <span>Applicant Age & Demographics</span>
                    </div>
                    <div class="nl-nudge-banner" style="margin-top:8px;"><i class="fa-solid fa-keyboard text-amber"></i> Please type your age in the chatbox below for instant pre-approval decision!</div>
                `;
                dynamicInputArea.appendChild(container);
                break;
        }
    }

    function setupPills(containerId, callback) {
        const el = document.getElementById(containerId);
        if (!el) return;
        el.querySelectorAll('.pill-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                el.querySelectorAll('.pill-btn').forEach(b => b.classList.remove('active'));
                const target = e.target.closest('.pill-btn');
                target.classList.add('active');
                callback(target.getAttribute('data-val'));
            });
        });
    }

    async function checkEarlyExitAndProceed(nextStepNumber) {
        // Only attempt early exit evaluation if we have collected key financial facts!
        if (!state.formData.make_code || !state.formData.loan_amount || !state.formData.net_salary) {
            renderStepInput(nextStepNumber);
            return;
        }

        try {
            const res = await fetch('/api/check-early-exit', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    pincode: state.formData.pincode,
                    net_salary: state.formData.net_salary,
                    loan_amount: state.formData.loan_amount,
                    make_code: state.formData.make_code,
                    age: state.formData.age,
                    employment_type: state.formData.employment_type
                })
            });
            const data = await res.json();
            if (data.is_confident_early_exit) {
                appendBotMessage(`<div style="color:var(--emerald); background:rgba(16,185,129,0.1); padding:10px 14px; border-radius:8px; margin-bottom:8px;"><i class="fa-solid fa-bolt text-emerald"></i> <strong>Early Decision Cutoff Triggered!</strong> High confidence decision reached with current information.</div>`);
                await submitFinalEvaluation();
            } else {
                renderStepInput(nextStepNumber);
            }
        } catch (err) {
            renderStepInput(nextStepNumber);
        }
    }

    async function submitFinalEvaluation() {
        dynamicInputArea.innerHTML = `
            <div class="step-box" style="text-align:center; padding:30px;">
                <div class="spinner" style="margin:0 auto 15px auto;"></div>
                <h4 style="font-size:16px; margin-bottom:6px;">Evaluating Application...</h4>
                <p style="font-size:12px; color:var(--text-muted);">Running LightGBM model inference, calculating LTV & Debt-to-Income ratios...</p>
            </div>
        `;

        const payload = {
            full_name: state.formData.full_name || 'Applicant',
            phone_number: state.formData.phone_number || '9876543210',
            email_address: state.formData.email_address || 'applicant@example.com',
            gender: state.formData.gender || 'Male',
            age: state.formData.age || 32,
            qualifications: 'GRAD',
            employment_type: state.formData.employment_type || 'SAL',
            resident_type: state.formData.resident_type || 'O',
            pincode: state.formData.pincode || '500090',
            net_salary: state.formData.net_salary || 45000,
            past_loans_active: 'NO_PAST_LOANS',
            product_code: 'MC',
            make_code: state.formData.make_code || 'JUPITER',
            vehicle_name: state.formData.vehicle_name || 'TVS Jupiter 125',
            vehicle_price: state.formData.vehicle_price || 110000,
            loan_amount: state.formData.loan_amount || 85000,
            ltv: state.formData.ltv || 77.2
        };

        try {
            const res = await fetch('/api/evaluate-loan', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            renderDecisionCard(data);
        } catch (err) {
            console.error("Evaluation Error:", err);
            dynamicInputArea.innerHTML = `<div class="step-box" style="color:var(--rose);">Error evaluating loan. Please check server logs.</div>`;
        }
    }

    function renderDecisionCard(data) {
        const isApproved = data.decision === 'APPROVED';
        const cardClass = isApproved ? 'decision-approved' : 'decision-declined';
        const iconClass = isApproved ? 'fa-circle-check text-emerald' : 'fa-circle-xmark text-rose';

        const approvalScore = data.approval_score != null ? data.approval_score : 85.0;
        const ltvStr = (data.risk_factors && data.risk_factors.LTV) ? data.risk_factors.LTV :
            (data.applicant_summary ? data.applicant_summary.ltv + '%' : '—');
        const foirStr = (data.risk_factors && data.risk_factors.FOIR) ? data.risk_factors.FOIR : '—';
        const pdStr = data.probability_of_default != null ? (data.probability_of_default * 100).toFixed(2) : '—';
        const optThresh = data.optimal_threshold != null ? (data.optimal_threshold * 100).toFixed(2) + '%' : '—';

        // Always derive from applicant_summary (server always populates this)
        const summary = data.applicant_summary || {};
        const reqLoan = summary.loan_amount != null
            ? '₹' + Number(summary.loan_amount).toLocaleString('en-IN')
            : (data.loan_details && data.loan_details.requested_loan
                ? '₹' + Number(data.loan_details.requested_loan).toLocaleString('en-IN')
                : '—');
        const vehPrice = summary.vehicle_price != null
            ? '₹' + Number(summary.vehicle_price).toLocaleString('en-IN')
            : '—';
        const vehMake = (data.loan_details && data.loan_details.vehicle_make)
            ? data.loan_details.vehicle_make
            : (state.formData.make_code || '—');
        const maskedPin = (data.applicant && data.applicant.masked_pincode) ? data.applicant.masked_pincode : '—';
        const pinTier = (data.risk_factors && data.risk_factors.pincode_tier) ? data.risk_factors.pincode_tier : 'Urban';
        const estimatedEmi = summary.estimated_emi != null
            ? '₹' + Number(summary.estimated_emi).toLocaleString('en-IN') + '/mo'
            : '—';
        const nlExplanation = data.natural_language_explanation || null;
        const declineCodes = data.decline_codes || [];

        const areaStr = (!maskedPin || maskedPin === 'Not Provided' || maskedPin === 'Not Specified' || maskedPin === '—' || pinTier === 'Not Provided')
            ? 'Not Provided'
            : `${maskedPin} (${pinTier})`;

        dynamicInputArea.innerHTML = `
            <div class="decision-card ${cardClass} animate-fade-in">
                <div style="font-size:28px; margin-bottom:8px;"><i class="fa-solid ${iconClass}"></i></div>
                <h2 style="font-size:22px; font-weight:700; margin-bottom:4px;">Loan ${data.decision || 'APPROVED'}</h2>
                <div style="font-size:13px; color:var(--text-muted); margin-bottom:15px;">Application Ref: ${data.session_id || 'ABC-123'}</div>

                <div class="decision-details-grid">
                    <div class="detail-box">
                        <span class="detail-label">Approval Score</span>
                        <span class="detail-val ${isApproved ? 'text-emerald' : 'text-rose'}">${approvalScore}/100</span>
                    </div>
                    <div class="detail-box">
                        <span class="detail-label">Calculated LTV</span>
                        <span class="detail-val">${ltvStr}</span>
                    </div>
                    <div class="detail-box">
                        <span class="detail-label">FOIR / DTI</span>
                        <span class="detail-val">${foirStr}</span>
                    </div>
                </div>

                <div style="text-align:left; background:rgba(255,255,255,0.03); border-radius:10px; padding:12px; margin-top:15px; font-size:12px;">
                    <div style="color:var(--text-muted); line-height:1.8;">
                        • Default Probability (PD): <strong>${pdStr}%</strong> (Threshold: ${optThresh})<br>
                        • Vehicle: <strong>${vehMake}</strong> | On-road Price: <strong>${vehPrice}</strong><br>
                        • Requested Loan: <strong>${reqLoan}</strong> | LTV: <strong>${ltvStr}</strong><br>
                        • Estimated EMI: <strong>${estimatedEmi}</strong> | FOIR: <strong>${foirStr}</strong><br>
                        • Area: <strong>${areaStr}</strong>
                    </div>
                </div>

                ${nlExplanation ? `
                <div style="background:rgba(239,68,68,0.08); border:1px solid rgba(239,68,68,0.25); border-radius:10px; padding:13px 14px; margin-top:12px; text-align:left;">
                    <div style="font-size:11px; font-weight:700; color:var(--rose); letter-spacing:0.05em; margin-bottom:6px;"><i class="fa-solid fa-triangle-exclamation"></i> DECLINE EXPLANATION</div>
                    <p style="font-size:13px; color:var(--text-muted); line-height:1.6; margin:0 0 8px;">${nlExplanation}</p>
                    ${declineCodes.length > 0 ? `<div style="font-size:11px; color:rgba(239,68,68,0.9); font-family:monospace; margin-top:4px;">${declineCodes.map(c => `<div>• ${c}</div>`).join('')}</div>` : ''}
                </div>` : ''}
                <button id="btn-restart" class="btn-primary" style="margin-top:20px; width:100%; justify-content:center;"><i class="fa-solid fa-rotate-right"></i> Start New Application</button>
            </div>
        `;

        document.getElementById('btn-restart').addEventListener('click', () => { window.location.reload(); });

        appendBotMessage(`
            <div style="font-size:14px; font-weight:600; margin-bottom:4px;" class="${isApproved ? 'text-emerald' : 'text-rose'}">
                <i class="fa-solid ${iconClass}"></i> Loan ${data.decision || 'APPROVED'}
            </div>
            <p>Application processing complete! Decision score: <strong>${approvalScore}/100</strong>.</p>
        `);
    }

    // Conversational Generative AI Mode Listener (Single Input Field Architecture)
    let conversationalFacts = {};

    async function handleConversationalTurn() {
        const text = chatInputText.value.trim();
        if (!text) return;

        appendUserMessage(text);
        chatInputText.value = '';

        try {
            const res = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    user_message: text,
                    collected_facts: conversationalFacts
                })
            });
            const data = await res.json();
            conversationalFacts = data.collected_facts || {};

            // Synchronize extracted facts with UI state
            if (conversationalFacts.make_code) state.formData.make_code = conversationalFacts.make_code;
            if (conversationalFacts.vehicle_name) state.formData.vehicle_name = conversationalFacts.vehicle_name;
            if (conversationalFacts.suggested_price) state.formData.suggested_price = parseFloat(conversationalFacts.suggested_price);
            if (conversationalFacts.vehicle_price) state.formData.vehicle_price = parseFloat(conversationalFacts.vehicle_price);
            if (conversationalFacts.loan_amount) state.formData.loan_amount = parseFloat(conversationalFacts.loan_amount);
            if (conversationalFacts.net_salary) state.formData.net_salary = parseFloat(conversationalFacts.net_salary);
            if (conversationalFacts.employment_type) state.formData.employment_type = conversationalFacts.employment_type;
            if (conversationalFacts.resident_type) state.formData.resident_type = conversationalFacts.resident_type;
            if (conversationalFacts.pincode) state.formData.pincode = String(conversationalFacts.pincode);
            if (conversationalFacts.age) state.formData.age = parseInt(conversationalFacts.age);

            if (data.is_complete && data.decision_result) {
                renderDecisionCard(data.decision_result);
            } else {
                appendBotMessage(`<p>${data.bot_message}</p>`);
                
                // Auto-advance step cards in dynamic input area to match next_feature exactly
                if (data.next_feature === 'make_code') {
                    renderStepInput(2);
                } else if (data.next_feature === 'vehicle_price') {
                    renderStepInput(3);
                } else if (data.next_feature === 'loan_amount') {
                    renderStepInput(4);
                } else if (data.next_feature === 'net_salary') {
                    renderStepInput(5);
                } else if (data.next_feature === 'employment_type') {
                    renderStepInput(6);
                } else if (data.next_feature === 'resident_type') {
                    renderStepInput(7);
                } else if (data.next_feature === 'pincode') {
                    renderStepInput(8);
                } else if (data.next_feature === 'age') {
                    renderStepInput(9);
                }
            }
        } catch (err) {
            appendBotMessage('<p style="color:var(--rose);">Error communicating with Conversational Decision Engine.</p>');
        }
    }

    btnSendChat.addEventListener('click', handleConversationalTurn);
    chatInputText.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') handleConversationalTurn();
    });

    // Audit Log Modal Event Handlers
    if (toggleAuditBtn && auditModal && closeAuditBtn) {
        toggleAuditBtn.addEventListener('click', async () => {
            auditModal.classList.remove('hidden');
            await fetchAuditLogs();
        });

        closeAuditBtn.addEventListener('click', () => {
            auditModal.classList.add('hidden');
        });

        auditModal.addEventListener('click', (e) => {
            if (e.target === auditModal) {
                auditModal.classList.add('hidden');
            }
        });
    }

    async function fetchAuditLogs() {
        if (!auditLogsContainer) return;
        auditLogsContainer.innerHTML = '<div style="color:var(--text-muted); text-align:center; padding:20px;"><i class="fa-solid fa-spinner fa-spin"></i> Loading audit logs...</div>';
        try {
            const res = await fetch('/api/audit-logs');
            const data = await res.json();
            if (!data.logs || data.logs.length === 0) {
                auditLogsContainer.innerHTML = '<div style="color:var(--text-muted); text-align:center; padding:20px;">No audit log records found yet. Submit a loan application to view logged entries.</div>';
                return;
            }
            auditLogsContainer.innerHTML = data.logs.map(log => `
                <div class="log-entry">
                    <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                        <span class="log-session">${log.session_id || 'ABC-LOG'}</span>
                        <span style="color:var(--text-dark);">${log.timestamp || ''}</span>
                    </div>
                    <div style="color:var(--text-muted);">
                        <strong>Applicant:</strong> ${log.applicant?.masked_name || 'N/A'} | ${log.applicant?.gender || ''}, Age ${log.applicant?.age || ''} | Pincode: ${log.applicant?.masked_pincode || ''}<br>
                        <strong>Financials:</strong> Salary: ₹${(log.financials?.net_salary || 0).toLocaleString()} | Loan: ₹${(log.financials?.loan_amount || 0).toLocaleString()} | Vehicle Price: ₹${(log.financials?.vehicle_price || 0).toLocaleString()} | LTV: ${log.financials?.calculated_ltv}%<br>
                        <strong>Decision:</strong> <span class="${log.ml_decisioning?.decision === 'APPROVED' ? 'text-emerald' : 'text-rose'}" style="font-weight:700;">${log.ml_decisioning?.decision || 'N/A'}</span> (Score: ${log.ml_decisioning?.approval_score || 'N/A'}/100, PD: ${((log.ml_decisioning?.model_probability_default || 0)*100).toFixed(2)}%)
                        ${log.ml_decisioning?.decline_codes?.length ? `<br><strong style="color:var(--rose);">Decline Reasons:</strong> ${log.ml_decisioning.decline_codes.join(' | ')}` : ''}
                    </div>
                </div>
            `).join('');
        } catch (err) {
            auditLogsContainer.innerHTML = '<div style="color:var(--rose); padding:10px;">Failed to load audit logs.</div>';
        }
    }
});

