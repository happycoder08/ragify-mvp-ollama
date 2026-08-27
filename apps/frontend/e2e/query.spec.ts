import { test, expect } from '@playwright/test';

// Helper to seed a valid, non-expired JWT for auth guards and API calls
async function seedJwt(page: import('@playwright/test').Page) {
  await page.addInitScript(() => {
    const payload = {
      sub: 'e2e-user',
      tenant_id: 'e2e-tenant',
      // Far future expiration (year 2100)
      exp: 4102444800,
    };
    const base64 = btoa(JSON.stringify(payload));
    const token = ['{}', base64, 'sig'].join('.');
    window.localStorage.setItem('ragify_jwt', token);
  });
}

// Helper to mock documents list
async function mockDocuments(page: import('@playwright/test').Page, documents: any[] = []) {
  await page.route('**/api/documents', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ documents }),
    });
  });
}

// Helper to mock /api/query with a given final payload and optional delay
async function mockQueryFinalOnce(page: import('@playwright/test').Page, finalPayload: any, delayMs = 50) {
  await page.route('**/api/query', async (route) => {
    // Simulate a bit of "streaming" latency before final arrives
    await new Promise((resolve) => setTimeout(resolve, delayMs));
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(finalPayload),
    });
  });
}

test.describe('Query page SSE-like flow', () => {
  test('Streaming completes: final answer persists and streaming stops', async ({ page }) => {
    await seedJwt(page);
    await mockDocuments(page, []);

    await mockQueryFinalOnce(page, {
      answer: 'Final streamed answer',
      refused: false,
      evidence: [
        { chunk_id: 'c1', snippet: 'snippet one', heading: 'Heading 1' },
      ],
      sources: [
        { filename: 'doc.txt' },
      ],
    });

    await page.goto('/query');

    const textarea = page.getByPlaceholder('What is the company vacation policy?');
    await textarea.fill('What is the policy?');

    const askButton = page.getByRole('button', { name: /Ask|Asking/i });
    await askButton.click();

    // While request is in flight, we should be "Asking..." (streaming state)
    await expect(page.getByRole('button', { name: /Asking.../i })).toBeVisible();

    // Before final, the answer-mode badge should not be visible
    await expect(page.locator('text=EXTRACTED')).toHaveCount(0);
    await expect(page.locator('text=CITED')).toHaveCount(0);
    await expect(page.locator('text=NOT FOUND')).toHaveCount(0);

    // After the final payload, the answer and badge should be visible and streaming indicator gone
    await expect(page.getByText('Final streamed answer')).toBeVisible();
    await expect(page.locator('text=EXTRACTED')).toHaveCount(1);
    await expect(page.locator('.cursor')).toHaveCount(0);
    await expect(page.getByRole('button', { name: 'Cancel' })).toHaveCount(0);
  });

  test('Cancel stops streaming and prevents final answer from appearing', async ({ page }) => {
    await seedJwt(page);
    await mockDocuments(page, []);

    // Simulate a slow backend; fulfill after a noticeable delay
    await page.route('**/api/query', async (route) => {
      await new Promise((resolve) => setTimeout(resolve, 1500));
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          answer: 'Answer that should not appear',
          refused: false,
          evidence: [
            { chunk_id: 'c1', snippet: 'late snippet' },
          ],
          sources: [
            { filename: 'late-doc.txt' },
          ],
        }),
      });
    });

    await page.goto('/query');

    const textarea = page.getByPlaceholder('What is the company vacation policy?');
    await textarea.fill('Should be cancelled');

    const askButton = page.getByRole('button', { name: /Ask|Asking/i });
    await askButton.click();

    // Wait for streaming state and then cancel promptly
    const cancelButton = page.getByRole('button', { name: 'Cancel' });
    await expect(cancelButton).toBeVisible();
    await cancelButton.click();

    // After cancel, streaming indicator and Cancel button should disappear
    await expect(page.locator('.cursor')).toHaveCount(0);
    await expect(page.getByRole('button', { name: 'Cancel' })).toHaveCount(0);

    // Give backend a bit of time to "respond" and ensure the late answer does not land in UI
    await page.waitForTimeout(2000);
    await expect(page.getByText('Answer that should not appear')).toHaveCount(0);
  });

  test('Doc scoping adds doc_ids to query payload', async ({ page }) => {
    await seedJwt(page);

    const documents = [
      {
        id: 1,
        filename: 'DocOne.txt',
        status: 'indexed',
        created_at: '2024-01-01T00:00:00Z',
        updated_at: '2024-01-01T00:00:00Z',
      },
      {
        id: 2,
        filename: 'DocTwo.txt',
        status: 'indexed',
        created_at: '2024-01-01T00:00:00Z',
        updated_at: '2024-01-01T00:00:00Z',
      },
    ];

    await mockDocuments(page, documents);

    let lastRequestBody: any | undefined;
    await page.route('**/api/query', async (route) => {
      const bodyText = route.request().postData() || '{}';
      lastRequestBody = JSON.parse(bodyText);
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          answer: 'Scoped answer',
          refused: false,
          evidence: [],
          sources: [],
        }),
      });
    });

    await page.goto('/query');

    // Switch to "Selected docs" scope
    const selectedDocsRadio = page.getByLabel('Selected docs');
    await selectedDocsRadio.check();

    // Select the first document in the multi-select
    const select = page.locator('select[multiple]');
    await expect(select).toBeVisible();
    await select.selectOption({ label: 'DocOne.txt' });

    const textarea = page.getByPlaceholder('What is the company vacation policy?');
    await textarea.fill('Scoped question');

    const askButton = page.getByRole('button', { name: /Ask|Asking/i });
    await askButton.click();

    // Wait until the intercepted request body has been captured
    await expect.poll(() => lastRequestBody).not.toBeUndefined();

    expect(Array.isArray(lastRequestBody!.doc_ids)).toBe(true);
    expect(lastRequestBody!.doc_ids).toEqual([1]);
  });
});
