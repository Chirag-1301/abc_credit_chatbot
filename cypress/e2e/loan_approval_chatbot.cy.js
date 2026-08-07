describe('ABC Credit Instant Loan Approval Engine & AI Chatbot Demo Suite', () => {

  // Demo speed configuration: 2.5s pause after each step for speaking over the presentation
  const STEP_PAUSE_MS = 2500;
  const TYPE_SPEED_MS = 75;

  beforeEach(() => {
    // Intercept backend API call
    cy.intercept('POST', '/api/chat').as('chatTurn');
    
    // Visit application home page
    cy.visit('/');
    // Verify Groq LLM is connected on startup
    cy.get('#groq-status-text').should('contain.text', 'Groq LLM Connected');
    cy.wait(1500); // Initial pause for audience intro
  });

  it('Demo Case 1: Early Pre-Approval -> Conversational Income Correction (10k) -> Final Decline Flow', () => {
    // Step 1: Applicant Name & Gender
    cy.get('#chat-input-text').type('I am Rahul Sharma, Male', { delay: TYPE_SPEED_MS });
    cy.get('#btn-send-chat').click();
    cy.wait('@chatTurn');
    cy.wait(STEP_PAUSE_MS);

    // Step 2: Vehicle Model (Honda Shine)
    cy.get('#chat-input-text').type('Honda Shine', { delay: TYPE_SPEED_MS });
    cy.get('#btn-send-chat').click();
    cy.wait('@chatTurn');
    cy.wait(STEP_PAUSE_MS);

    // Step 3: Negotiated Vehicle Price (110k)
    cy.get('#chat-input-text').type('110k', { delay: TYPE_SPEED_MS });
    cy.get('#btn-send-chat').click();
    cy.wait('@chatTurn');
    cy.wait(STEP_PAUSE_MS);

    // Step 4: Requested Loan Amount (80k)
    cy.get('#chat-input-text').type('80k', { delay: TYPE_SPEED_MS });
    cy.get('#btn-send-chat').click();
    cy.wait('@chatTurn');
    cy.wait(STEP_PAUSE_MS);

    // Step 5: Net Monthly Salary (75k - High Salary triggers Early Exit Pre-Approval)
    cy.get('#chat-input-text').type('75k', { delay: TYPE_SPEED_MS });
    cy.get('#btn-send-chat').click();
    cy.wait('@chatTurn');

    // Verify Early Pre-Approval Decision Card
    cy.contains('Loan APPROVED', { timeout: 15000 }).scrollIntoView().should('exist');
    cy.wait(4500); // Pause to point out early approval card to class

    // Step 6: Interactive Correction - User corrects salary to 10k per month
    cy.get('#chat-input-text').type('Wait, I made a mistake, my actual net monthly salary is only 10k', { delay: TYPE_SPEED_MS });
    cy.get('#btn-send-chat').click();
    cy.wait('@chatTurn');
    cy.contains('.message-bot', 'employment sector', { timeout: 10000 }).should('exist');
    cy.wait(STEP_PAUSE_MS);

    // Step 7: Employment Sector (Freelancer / Gig worker)
    cy.get('#chat-input-text').type('Freelancer gig worker', { delay: TYPE_SPEED_MS });
    cy.get('#btn-send-chat').click();
    cy.wait('@chatTurn');
    cy.contains('.message-bot', 'residential status', { timeout: 10000 }).should('exist');
    cy.wait(STEP_PAUSE_MS);

    // Step 8: Residential Status (Rented)
    cy.get('#chat-input-text').type('Rented house', { delay: TYPE_SPEED_MS });
    cy.get('#btn-send-chat').click();
    cy.wait('@chatTurn');
    cy.contains('.message-bot', 'Pincode', { timeout: 10000 }).should('exist');
    cy.wait(STEP_PAUSE_MS);

    // Step 9: Pincode (517589 - High Risk Area)
    cy.get('#chat-input-text').type('517589', { delay: TYPE_SPEED_MS });
    cy.get('#btn-send-chat').click();
    cy.wait('@chatTurn');
    cy.contains('.message-bot', 'age', { timeout: 10000 }).should('exist');
    cy.wait(STEP_PAUSE_MS);

    // Step 10: Age (24)
    cy.get('#chat-input-text').type('24', { delay: TYPE_SPEED_MS });
    cy.get('#btn-send-chat').click();
    cy.wait('@chatTurn');

    // Verify Final Decision Card - DECLINED due to 10k income correction & risk profile
    cy.contains('Loan DECLINED', { timeout: 15000 }).scrollIntoView().should('exist');
    cy.wait(5000); // Finale pause to explain dynamic re-evaluation to class
  });

  it('Demo Case 2: High-Risk / High-Leverage Decline with Adverse Action Transparency (Direct Decline Path)', () => {
    // Step 1: Applicant Name & Gender
    cy.get('#chat-input-text').type('Chirag Patel, Male', { delay: TYPE_SPEED_MS });
    cy.get('#btn-send-chat').click();
    cy.wait('@chatTurn');
    cy.wait(STEP_PAUSE_MS);

    // Step 2: Vehicle Model (TVS Apache)
    cy.get('#chat-input-text').type('TVS Apache', { delay: TYPE_SPEED_MS });
    cy.get('#btn-send-chat').click();
    cy.wait('@chatTurn');
    cy.wait(STEP_PAUSE_MS);

    // Step 3: Negotiated Vehicle Price (140k)
    cy.get('#chat-input-text').type('140k', { delay: TYPE_SPEED_MS });
    cy.get('#btn-send-chat').click();
    cy.wait('@chatTurn');
    cy.wait(STEP_PAUSE_MS);

    // Step 4: Requested Loan Amount (125k)
    cy.get('#chat-input-text').type('125k', { delay: TYPE_SPEED_MS });
    cy.get('#btn-send-chat').click();
    cy.wait('@chatTurn');
    cy.wait(STEP_PAUSE_MS);

    // Step 5: Net Monthly Salary (18k)
    cy.get('#chat-input-text').type('18k', { delay: TYPE_SPEED_MS });
    cy.get('#btn-send-chat').click();
    cy.wait('@chatTurn');
    cy.wait(STEP_PAUSE_MS);

    // Step 6: Employment Sector (Freelancer)
    cy.get('#chat-input-text').type('Freelancer gig worker', { delay: TYPE_SPEED_MS });
    cy.get('#btn-send-chat').click();
    cy.wait('@chatTurn');
    cy.wait(STEP_PAUSE_MS);

    // Step 7: Residential Status (Rented)
    cy.get('#chat-input-text').type('Rented house', { delay: TYPE_SPEED_MS });
    cy.get('#btn-send-chat').click();
    cy.wait('@chatTurn');
    cy.wait(STEP_PAUSE_MS);

    // Step 8: Pincode (517589 - High Risk Area)
    cy.get('#chat-input-text').type('517589', { delay: TYPE_SPEED_MS });
    cy.get('#btn-send-chat').click();
    cy.wait('@chatTurn');
    cy.wait(STEP_PAUSE_MS);

    // Step 9: Age (24)
    cy.get('#chat-input-text').type('24', { delay: TYPE_SPEED_MS });
    cy.get('#btn-send-chat').click();
    cy.wait('@chatTurn');

    // Verify Final Decision Card - DECLINED
    cy.contains('Loan DECLINED', { timeout: 15000 }).scrollIntoView().should('exist');
    cy.wait(5000); // Finale pause to explain adverse action codes to class
  });

});
